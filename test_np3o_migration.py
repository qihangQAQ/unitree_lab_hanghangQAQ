#!/usr/bin/env python3
"""
测试 NP3O 迁移是否成功
"""

import sys
import os

# 添加项目路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(project_root, "source/unitree_rl_lab"))

def test_basic_injection():
    """测试基本类注入"""
    print("=" * 60)
    print("测试 NP3O 类注入")
    print("=" * 60)

    # 1. 导入 rsl_rl_ext 扩展包
    try:
        import unitree_rl_lab.rsl_rl_ext
        print("✅ 成功导入 unitree_rl_lab.rsl_rl_ext")
    except Exception as e:
        print(f"❌ 导入 unitree_rl_lab.rsl_rl_ext 失败: {e}")
        return False

    # 2. 检查 rsl_rl 模块中是否存在 NP3O 类
    try:
        import rsl_rl
        print("✅ 成功导入 rsl_rl")
    except Exception as e:
        print(f"❌ 导入 rsl_rl 失败: {e}")
        return False

    # 检查顶级模块
    classes_to_check = [
        ("NP3O", "算法类"),
        ("ActorCriticNP3O", "Actor-Critic 类"),
        ("NP3ORolloutStorage", "存储类"),
        ("NP3ORunner", "Runner 类"),
    ]

    all_passed = True
    for class_name, description in classes_to_check:
        if hasattr(rsl_rl, class_name):
            print(f"✅ rsl_rl.{class_name} 存在 ({description})")
        else:
            print(f"❌ rsl_rl.{class_name} 不存在 ({description})")
            all_passed = False

    # 检查子模块
    submodules_to_check = [
        ("algorithms", "NP3O", "算法子模块"),
        ("modules", "ActorCriticNP3O", "模块子模块"),
        ("storage", "NP3ORolloutStorage", "存储子模块"),
        ("runners", "NP3ORunner", "Runner子模块"),
    ]

    for submodule_name, class_name, description in submodules_to_check:
        if hasattr(rsl_rl, submodule_name):
            submodule = getattr(rsl_rl, submodule_name)
            if hasattr(submodule, class_name):
                print(f"✅ rsl_rl.{submodule_name}.{class_name} 存在 ({description})")
            else:
                print(f"❌ rsl_rl.{submodule_name}.{class_name} 不存在 ({description})")
                all_passed = False
        else:
            print(f"❌ rsl_rl.{submodule_name} 子模块不存在")
            all_passed = False

    return all_passed

def test_eval_class_name():
    """测试 eval() 能否解析 class_name 字符串"""
    print("\n" + "=" * 60)
    print("测试 class_name 解析")
    print("=" * 60)

    import rsl_rl

    # 模拟配置文件中使用的 class_name
    class_names = [
        "ActorCriticNP3O",
        "NP3O",
        "NP3ORunner",
        "NP3ORolloutStorage",
    ]

    all_passed = True
    for class_name in class_names:
        try:
            # 模拟 rsl_rl 中使用的 eval()
            cls = eval(class_name, vars(rsl_rl))
            print(f"✅ eval('{class_name}') 成功 -> {cls}")
        except Exception as e:
            print(f"❌ eval('{class_name}') 失败: {e}")
            all_passed = False

    # 测试在子模块上下文中 eval
    contexts = [
        ("rsl_rl.modules", ["ActorCriticNP3O"]),
        ("rsl_rl.algorithms", ["NP3O"]),
        ("rsl_rl.runners", ["NP3ORunner"]),
        ("rsl_rl.storage", ["NP3ORolloutStorage"]),
    ]

    for module_path, names in contexts:
        try:
            module = eval(module_path)
            for name in names:
                if hasattr(module, name):
                    print(f"✅ {module_path}.{name} 可访问")
                else:
                    print(f"❌ {module_path}.{name} 不可访问")
                    all_passed = False
        except Exception as e:
            print(f"❌ 访问 {module_path} 失败: {e}")
            all_passed = False

    return all_passed

def test_config_loading():
    """测试配置文件加载"""
    print("\n" + "=" * 60)
    print("测试配置文件加载")
    print("=" * 60)

    try:
        # 导入配置文件
        sys.path.insert(0, os.path.join(project_root, "source/unitree_rl_lab/unitree_rl_lab"))
        from tasks.locomotion.agents.rsl_rl_np3o_cfg import UnitreeNp3oRunnerCfg

        print("✅ 成功导入 UnitreeNp3oRunnerCfg")

        # 检查配置中的 class_name
        cfg = UnitreeNp3oRunnerCfg()

        print(f"配置信息:")
        print(f"  - runner class_name: {getattr(cfg, 'class_name', '未设置')}")
        print(f"  - policy class_name: {cfg.policy.class_name}")

        # 注意：algorithm 的 class_name 在配置中被注释了
        # print(f"  - algorithm class_name: {cfg.algorithm.class_name}")

        return True
    except Exception as e:
        print(f"❌ 配置文件加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("NP3O 迁移测试开始")
    print(f"Python 路径: {sys.executable}")
    print(f"工作目录: {os.getcwd()}")
    print(f"项目根目录: {project_root}")

    results = []

    # 测试基本注入
    results.append(("基本类注入", test_basic_injection()))

    # 测试 eval 解析
    results.append(("class_name 解析", test_eval_class_name()))

    # 测试配置文件加载
    results.append(("配置文件加载", test_config_loading()))

    # 汇总结果
    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)

    all_passed = True
    for test_name, passed in results:
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"{test_name}: {status}")
        if not passed:
            all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 所有测试通过！NP3O 迁移成功。")
        print("使用 NP3O 时，请在训练脚本开头添加:")
        print("    import unitree_rl_lab.rsl_rl_ext")
    else:
        print("⚠️  部分测试失败，请检查以上错误信息。")

    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)