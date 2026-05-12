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

# 延迟导入 rsl_rl，在 inject_classes() 中处理
rsl_rl = None

# 导入本地 NP3O 类
from .algorithms.np3o import NP3O
from .modules.actor_critic_np3o import ActorCriticNP3O
from .storage.rollout_storage_np3o import NP3ORolloutStorage
from .runners.on_policy_runner_np3o import NP3ORunner
from .modules.actor_critic_perception import ActorCriticPerception
from .modules.actor_critic_depth import ActorCriticDepth

# 定义要注入的类映射
_CLASSES_TO_INJECT = {
    'NP3O': NP3O,
    'ActorCriticNP3O': ActorCriticNP3O,
    'NP3ORolloutStorage': NP3ORolloutStorage,
    'NP3ORunner': NP3ORunner,
    'ActorCriticPerception': ActorCriticPerception,
    'ActorCriticDepth': ActorCriticDepth,
}

def inject_classes():
    """将 NP3O 类注入到 rsl_rl 模块的命名空间中"""
    print(f"🔍 开始注入类到 rsl_rl 模块...")

    global rsl_rl
    # 尝试导入 rsl_rl 模块
    try:
        import rsl_rl as rsl_rl_module
        rsl_rl = rsl_rl_module
        version = getattr(rsl_rl, '__version__', '未知')
        print(f"✅ 成功导入 rsl_rl 模块，版本: {version}")
    except ImportError as e:
        warnings.warn(f"无法导入 rsl_rl 模块: {e}")
        print(f"❌ rsl_rl 模块导入失败")
        return False

    try:
        # 1. 注入到 rsl_rl 顶级模块
        for class_name, cls in _CLASSES_TO_INJECT.items():
            setattr(rsl_rl, class_name, cls)

        # 2. 导入并注入到相应的子模块中
        # NP3O -> rsl_rl.algorithms
        try:
            import rsl_rl.algorithms
            rsl_rl.algorithms.NP3O = NP3O
            print(f"   ✅ 成功注入 NP3O 到 rsl_rl.algorithms")
        except ImportError as e:
            print(f"   ⚠️  无法导入 rsl_rl.algorithms: {e}")

        # ActorCriticNP3O 和 ActorCriticPerception, ActorCriticDepth -> rsl_rl.modules
        try:
            import rsl_rl.modules
            rsl_rl.modules.ActorCriticNP3O = ActorCriticNP3O
            rsl_rl.modules.ActorCriticPerception = ActorCriticPerception
            rsl_rl.modules.ActorCriticDepth = ActorCriticDepth
            print(f"   ✅ 成功注入 ActorCriticNP3O, ActorCriticPerception, ActorCriticDepth 到 rsl_rl.modules")
        except ImportError as e:
            print(f"   ⚠️  无法导入 rsl_rl.modules: {e}")

        # NP3ORolloutStorage -> rsl_rl.storage
        try:
            import rsl_rl.storage
            rsl_rl.storage.NP3ORolloutStorage = NP3ORolloutStorage
            print(f"   ✅ 成功注入 NP3ORolloutStorage 到 rsl_rl.storage")
        except ImportError as e:
            print(f"   ⚠️  无法导入 rsl_rl.storage: {e}")

        # NP3ORunner -> rsl_rl.runners
        try:
            import rsl_rl.runners
            rsl_rl.runners.NP3ORunner = NP3ORunner
            print(f"   ✅ 成功注入 NP3ORunner 到 rsl_rl.runners")

            # 尝试将 ActorCriticPerception 和 ActorCriticDepth 添加到 on_policy_runner 模块的全局命名空间
            try:
                import rsl_rl.runners.on_policy_runner
                rsl_rl.runners.on_policy_runner.ActorCriticPerception = ActorCriticPerception
                rsl_rl.runners.on_policy_runner.ActorCriticDepth = ActorCriticDepth
                print(f"   ✅ 成功注入 ActorCriticPerception, ActorCriticDepth 到 rsl_rl.runners.on_policy_runner")
            except ImportError as e:
                print(f"   ⚠️  无法导入 rsl_rl.runners.on_policy_runner: {e}")
        except ImportError as e:
            print(f"   ⚠️  无法导入 rsl_rl.runners: {e}")

        # 3. 确保这些类也在子模块的 __all__ 列表中（如果存在）
        try:
            _update_submodule_all('algorithms', ['NP3O'])
        except Exception as e:
            print(f"   ⚠️  更新 algorithms.__all__ 时出错: {e}")

        try:
            _update_submodule_all('modules', ['ActorCriticNP3O', 'ActorCriticPerception', 'ActorCriticDepth'])
        except Exception as e:
            print(f"   ⚠️  更新 modules.__all__ 时出错: {e}")

        try:
            _update_submodule_all('storage', ['NP3ORolloutStorage'])
        except Exception as e:
            print(f"   ⚠️  更新 storage.__all__ 时出错: {e}")

        try:
            _update_submodule_all('runners', ['NP3ORunner', 'ActorCriticPerception', 'ActorCriticDepth'])
        except Exception as e:
            print(f"   ⚠️  更新 runners.__all__ 时出错: {e}")

        print("✅ NP3O 和 Perception 类已成功注入到 rsl_rl 模块中")
        print(f"   可用类: {list(_CLASSES_TO_INJECT.keys())}")
        return True

    except Exception as e:
        warnings.warn(f"注入 NP3O 类时出错: {e}")
        print(f"❌ 注入过程中出错: {e}")
        import traceback
        traceback.print_exc()
        return False

def _update_submodule_all(submodule_name, class_names):
    """更新子模块的 __all__ 列表（如果存在）"""
    try:
        # 动态导入子模块
        submodule = __import__(f'rsl_rl.{submodule_name}', fromlist=[''])
        if hasattr(submodule, '__all__'):
            # 确保类名在 __all__ 中
            current_all = list(submodule.__all__)
            for class_name in class_names:
                if class_name not in current_all:
                    current_all.append(class_name)
            submodule.__all__ = current_all
            print(f"   ✅ 更新了 {submodule_name}.__all__: {class_names}")
    except ImportError as e:
        print(f"   ⚠️  无法导入 rsl_rl.{submodule_name} 来更新 __all__: {e}")

# 自动注入
_injection_successful = inject_classes()

# 导出 NP3O 类
__all__ = ['NP3O', 'ActorCriticNP3O', 'ActorCriticPerception', 'ActorCriticDepth', 'NP3ORolloutStorage', 'NP3ORunner', 'inject_classes']

# 重新导出以便直接导入
from .algorithms import NP3O
from .modules import ActorCriticNP3O, ActorCriticPerception, ActorCriticDepth
from .storage import NP3ORolloutStorage
from .runners import NP3ORunner