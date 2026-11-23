"""Persistent Isaac-side ROS 2 topic adapter for visual pick-and-place.

Uses standard ROS messages: /pick_place_goal, /pick_status, and /pick_cancel.
"""

from pathlib import Path as _ProjectPath
import sys as _sys
PROJECT_ROOT = _ProjectPath(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "src") not in _sys.path:
    _sys.path.insert(0, str(PROJECT_ROOT / "src"))

import asyncio
import builtins
import json
import math
import queue
import runpy
import threading
import time
import uuid
from pathlib import Path

import omni.kit.app
import omni.timeline
import rclpy
from geometry_msgs.msg import PointStamped
from rclpy.context import Context
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import String


ADAPTER_KEY = '_isaac_ros2_pick_adapter'
VISUAL_PICK_SCRIPT = PROJECT_ROOT / 'src/bottle_sim/visual_pick.py'


class PickTaskAdapter:
    def __init__(self):
        self.context = Context()
        rclpy.init(context=self.context)
        self.node = Node('isaac_pick_task_adapter', context=self.context)
        self.publisher = self.node.create_publisher(String, '/pick_status', 10)
        self.goal_subscription = self.node.create_subscription(
            PointStamped, '/pick_place_goal', self.on_target, 10)
        self.cancel_subscription = self.node.create_subscription(
            String, '/pick_cancel', self.on_cancel, 10)
        self.executor = SingleThreadedExecutor(context=self.context)
        self.executor.add_node(self.node)
        self.pending = queue.Queue(maxsize=1)
        self.lock = threading.Lock()
        self.busy = False
        self.current_task_id = None
        self.cancel_event = threading.Event()
        self.worker = threading.Thread(target=self.executor.spin,
                                       name='isaac-pick-ros2', daemon=True)
        self.monitor = None
        self.started_at = 0.0

    def publish_status(self, status, detail='', task_id=None, **fields):
        payload = {'status': status, 'detail': detail, **fields}
        if task_id is not None:
            payload['task_id'] = task_id
        message = String()
        message.data = json.dumps(payload, ensure_ascii=False)
        self.publisher.publish(message)
        print('[ROS2 pick] ' + message.data)

    def on_target(self, msg):
        point = [msg.point.x, msg.point.y, msg.point.z]
        if msg.header.frame_id != 'world':
            self.publish_status('rejected', 'frame_id must be world')
            return
        if not all(math.isfinite(value) for value in point):
            self.publish_status('rejected', 'coordinates must be finite')
            return
        if abs(msg.point.z) > 1e-6:
            self.publish_status('rejected',
                                'point.z must be 0; placement height is derived from the table')
            return

        with self.lock:
            if self.busy:
                self.publish_status('rejected', 'another pick-and-place task is active')
                return
            task_id = uuid.uuid4().hex
            self.busy = True
            self.current_task_id = task_id
            self.cancel_event.clear()
            self.pending.put_nowait((point[:2], task_id))
        self.publish_status('accepted', 'visual pick-and-place goal accepted',
                            task_id=task_id, place_target_world_xy_m=point[:2])

    def on_cancel(self, msg):
        try:
            request = json.loads(msg.data)
            task_id = str(request.get('task_id', ''))
        except (TypeError, json.JSONDecodeError):
            self.publish_status('cancel_rejected', 'cancel request must be JSON with task_id')
            return
        with self.lock:
            if not self.busy or task_id != self.current_task_id:
                self.publish_status('cancel_rejected', 'task_id is not active', task_id=task_id)
                return
            self.cancel_event.set()
        self.publish_status('cancel_requested', 'cancellation will be handled at a safe update point',
                            task_id=task_id)

    def report_phase(self, task_id, phase, detail):
        self.publish_status('running', detail, task_id=task_id, phase=phase,
                            elapsed_seconds=max(0.0, time.monotonic()-self.started_at))

    def release_goal(self, task_id):
        with self.lock:
            if self.current_task_id == task_id:
                self.busy = False
                self.current_task_id = None
                self.cancel_event.clear()

    async def run(self):
        try:
            while True:
                try:
                    target, task_id = self.pending.get_nowait()
                except queue.Empty:
                    await omni.kit.app.get_app().next_update_async()
                    continue

                self.started_at = time.monotonic()
                try:
                    timeline = omni.timeline.get_timeline_interface()
                    if timeline.is_playing():
                        self.release_goal(task_id)
                        self.publish_status('rejected',
                                            'pause or stop the simulation before sending a goal',
                                            task_id=task_id)
                        continue
                    visual = runpy.run_path(str(VISUAL_PICK_SCRIPT))
                    result = await visual['start_visual_pick'](
                        place_xy_world_m=target,
                        progress_callback=lambda phase, detail: self.report_phase(
                            task_id, phase, detail),
                        cancel_requested=self.cancel_event.is_set,
                    )
                    state = str(result.get('status', 'Unknown')).lower()
                    if state not in ('succeeded', 'failed', 'timedout', 'cancelled'):
                        state = 'failed'
                    self.release_goal(task_id)
                    self.publish_status(
                        state, str(result.get('detail', '')),
                        task_id=task_id,
                        phase=result.get('phase', ''),
                        place_target_world_xy_m=target,
                        place_target_m=result.get('place_target_m'),
                        final_position_m=result.get('final_position_m'),
                        max_hold_transport_slip_m=result.get('max_hold_transport_slip_m', 0.0),
                        metrics=result.get('metrics', {}),
                    )
                except asyncio.CancelledError:
                    self.release_goal(task_id)
                    self.publish_status('cancelled', 'pick-and-place task cancelled',
                                        task_id=task_id)
                    raise
                except Exception as exc:
                    terminal = 'cancelled' if self.cancel_event.is_set() else 'failed'
                    self.release_goal(task_id)
                    self.publish_status(terminal, str(exc), task_id=task_id,
                                        place_target_world_xy_m=target)
                finally:
                    self.release_goal(task_id)
                    self.pending.task_done()
        finally:
            self.executor.shutdown(timeout_sec=3.0)
            self.worker.join(timeout=3.0)
            if self.worker.is_alive():
                print('[ROS2 pick] Executor did not stop; node left intact for safe cleanup.')
            else:
                self.node.destroy_node()
                self.context.shutdown()

    def start(self):
        self.worker.start()
        self.monitor = asyncio.ensure_future(self.run())
        print('[ROS2 pick-place] Persistent topic adapter ready; waiting for goals.')


def start():
    previous = getattr(builtins, ADAPTER_KEY, None)
    if previous is not None and previous.monitor is not None and not previous.monitor.done():
        raise RuntimeError('ROS2 pick adapter is already running')
    adapter = PickTaskAdapter()
    setattr(builtins, ADAPTER_KEY, adapter)
    adapter.start()
    return adapter


if __name__ == '__main__':
    start()
