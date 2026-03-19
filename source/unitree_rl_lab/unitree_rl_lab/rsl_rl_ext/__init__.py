# NP3O 扩展包 - 猴子补丁注入脚本
"""
该模块将 NP3O 相关类注入到 rsl_rl 模块的命名空间中，
使现有配置文件中的 class_name 参数（如 "ActorCriticNP3O", "NP3O", "NP3ORunner"）
能够通过 rsl_rl 模块的 eval() 正确解析。

使用方法：
在训练脚本开头添加：import unitree_rl_lab.rsl_rl_ext
"""

import sys
import importlib
import warnings

# 尝试导入 rsl_rl 模块
try:
    import rsl_rl
except ImportError as e:
    warnings.warn(f"无法导入 rsl_rl 模块: {e}")
    rsl_rl = None

# 导入本地 NP3O 类
from .algorithms.np3o import NP3O
from .modules.actor_critic_np3o import ActorCriticNP3O
from .storage.rollout_storage_np3o import NP3ORolloutStorage
from .runners.on_policy_runner_np3o import NP3ORunner

# 定义要注入的类映射
_CLASSES_TO_INJECT = {
    'NP3O': NP3O,
    'ActorCriticNP3O': ActorCriticNP3O,
    'NP3ORolloutStorage': NP3ORolloutStorage,
    'NP3ORunner': NP3ORunner,
}

def inject_classes():
    """将 NP3O 类注入到 rsl_rl 模块的命名空间中"""
    if rsl_rl is None:
        warnings.warn("rsl_rl 模块未导入，无法注入 NP3O 类")
        return False

    try:
        # 1. 注入到 rsl_rl 顶级模块
        for class_name, cls in _CLASSES_TO_INJECT.items():
            setattr(rsl_rl, class_name, cls)

        # 2. 注入到相应的子模块中
        # NP3O -> rsl_rl.algorithms
        if hasattr(rsl_rl, 'algorithms'):
            rsl_rl.algorithms.NP3O = NP3O

        # ActorCriticNP3O -> rsl_rl.modules
        if hasattr(rsl_rl, 'modules'):
            rsl_rl.modules.ActorCriticNP3O = ActorCriticNP3O

        # NP3ORolloutStorage -> rsl_rl.storage
        if hasattr(rsl_rl, 'storage'):
            rsl_rl.storage.NP3ORolloutStorage = NP3ORolloutStorage

        # NP3ORunner -> rsl_rl.runners
        if hasattr(rsl_rl, 'runners'):
            rsl_rl.runners.NP3ORunner = NP3ORunner

        # 3. 确保这些类也在子模块的 __all__ 列表中（如果存在）
        _update_submodule_all('algorithms', ['NP3O'])
        _update_submodule_all('modules', ['ActorCriticNP3O'])
        _update_submodule_all('storage', ['NP3ORolloutStorage'])
        _update_submodule_all('runners', ['NP3ORunner'])

        print("✅ NP3O 类已成功注入到 rsl_rl 模块中")
        print(f"   可用类: {list(_CLASSES_TO_INJECT.keys())}")
        return True

    except Exception as e:
        warnings.warn(f"注入 NP3O 类时出错: {e}")
        return False

def _update_submodule_all(submodule_name, class_names):
    """更新子模块的 __all__ 列表（如果存在）"""
    if not hasattr(rsl_rl, submodule_name):
        return

    submodule = getattr(rsl_rl, submodule_name)
    if hasattr(submodule, '__all__'):
        # 确保类名在 __all__ 中
        current_all = list(submodule.__all__)
        for class_name in class_names:
            if class_name not in current_all:
                current_all.append(class_name)
        submodule.__all__ = current_all

# 自动注入
_injection_successful = inject_classes()

# 导出 NP3O 类
__all__ = ['NP3O', 'ActorCriticNP3O', 'NP3ORolloutStorage', 'NP3ORunner', 'inject_classes']

# 重新导出以便直接导入
from .algorithms import NP3O
from .modules import ActorCriticNP3O
from .storage import NP3ORolloutStorage
from .runners import NP3ORunner