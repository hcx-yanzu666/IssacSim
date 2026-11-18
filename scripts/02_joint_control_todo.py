"""旧版路线参考模板，不再作为第二课入口。

新教学方式请运行 02_read_joint_state.py，再逐课加入小功能。

目的：先理解生命周期，再逐步填入代码。
当前文件直接运行不会初始化物理或移动机器人。
"""

ROBOT_PATH = "/World/Franka"
CUBE_PATH = "/World/PracticeCube"


class JointControlLesson:
    """后续由仿真生命周期调用；不是执行一次便完成整个动作的脚本。"""

    def __init__(self):
        self.robot = None
        self.dof_indices = {}
        self.elapsed_seconds = 0.0

    async def initialize(self):
        # TODO 1：确认时间线状态，建立或复用 World / SimulationContext。
        # 编辑器内使用异步初始化，不能复制 standalone 示例另建 SimulationApp。
        # TODO 2：用 SingleArticulation(prim_path=ROBOT_PATH) 包装已有机械臂。
        # 这个 Python 包装器不代表重新创建一台机械臂。
        # TODO 3：物理初始化完成后，初始化 articulation 并读取 dof_names。
        # TODO 4：用 get_joint_positions() 读取实际位置，按名称建立 DOF 索引。
        # TODO 5：确认第二根手指是否存在 mimic 联动，不能凭名字猜控制方式。
        raise NotImplementedError("下一课实现：物理初始化与实际关节状态读取。")

    def on_physics_step(self, dt):
        # TODO 6：通过 add_physics_callback 注册物理步回调，dt 为仿真步长。
        # 不要在 UI 线程 while True，也不要用 time.sleep 控制运动。
        # TODO 7：以当前姿态为起点，随仿真时间平滑更新关节目标。
        # 通过 ArticulationAction + apply_action 给驱动器下达目标。
        # set_joint_positions 直接写状态，适合重置，不用它伪装连续物理运动。
        # TODO 8：依次执行准备姿态、张开、闭合，观察实际位置是否跟随目标。
        raise NotImplementedError("后续实现：每个物理步更新控制目标。")

    async def reset(self):
        # TODO 9：恢复机械臂初始关节位置，清零速度并同步 drive 目标。
        # 同时恢复方块初始位姿、线速度和角速度，重置任务计时与阶段。
        # TODO 10：处理 Stop/重新 Play 后物理句柄失效，必要时重新初始化。
        raise NotImplementedError("后续实现：可重复的场景重置。")

    def cleanup(self):
        # TODO 11：移除本课注册的物理步回调与事件订阅。
        # 重复运行脚本不能堆积多个控制器，同时向同一台机械臂下达指令。
        raise NotImplementedError("后续实现：控制器退出和重复运行清理。")


if __name__ == "__main__":
    print("这是关节控制 TODO 模板，尚未执行运动。请先完成 01_inspect_scene.py。")
