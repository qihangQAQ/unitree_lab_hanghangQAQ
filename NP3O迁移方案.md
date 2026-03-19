# NP3O 算法迁移方案

## 当前问题
NP3O 算法直接修改了 conda 环境中的 `rsl_rl` 包，破坏了环境隔离，不符合二次开发应在 `unitree_rl_lab` 中进行的架构原则。

## 迁移目标
1. 将 NP3O 实现从 `site-packages/rsl_rl` 移到 `unitree_rl_lab` 项目内
2. 通过运行时注入（猴子补丁）使 rsl_rl 能识别 NP3O 类
3. 清理 conda 环境中的修改，恢复原始状态
4. 确保现有配置文件无需修改

## 文件结构规划

```
unitree_rl_lab/
├── rsl_rl_ext/                    # NP3O 扩展包
│   ├── __init__.py               # 猴子补丁注入逻辑
│   ├── algorithms/
│   │   ├── __init__.py
│   │   └── np3o.py              # NP3O 算法核心
│   ├── modules/
│   │   ├── __init__.py
│   │   └── actor_critic_np3o.py # 带 cost critics 的 ActorCritic
│   ├── storage/
│   │   ├── __init__.py
│   │   └── rollout_storage_np3o.py # NP3O 专用存储
│   └── runners/
│       ├── __init__.py
│       └── on_policy_runner_np3o.py # NP3O 专用 runner
└── tasks/locomotion/agents/
    └── rsl_rl_np3o_cfg.py       # 现有配置文件（无需修改）
```

## 迁移步骤

### 步骤 1：创建扩展目录结构
在 `source/unitree_rl_lab/unitree_rl_lab/` 下创建 `rsl_rl_ext/` 目录及子目录。

### 步骤 2：复制 NP3O 实现文件
从 conda 环境的 `rsl_rl` 包中复制四个核心文件到新目录：
- `algorithms/np3o.py`
- `modules/actor_critic_np3o.py`
- `storage/rollout_storage_np3o.py`
- `runners/on_policy_runner_np3o.py`

### 步骤 3：编写猴子补丁脚本
创建 `rsl_rl_ext/__init__.py`，在导入时将 NP3O 类注入到 rsl_rl 模块的命名空间。

### 步骤 4：创建各子模块的 `__init__.py`
在每个子目录中创建 `__init__.py` 文件，正确导出相应的类。

### 步骤 5：清理 conda 环境修改
从 `site-packages/rsl_rl` 中移除 NP3O 相关修改：
1. 删除四个新增的 `.py` 文件
2. 从各 `__init__.py` 中移除 NP3O 的导入语句

### 步骤 6：更新导入机制
修改 `rsl_rl_ext/__init__.py` 确保：
1. NP3O 类被正确注入到 rsl_rl 模块
2. 现有配置文件的 `class_name` 参数（如 `"ActorCriticNP3O"`, `"NP3O"`, `"NP3ORunner"`）仍能正常工作

### 步骤 7：使用前导入扩展包
在训练脚本（如 `scripts/rsl_rl/train.py`）开头添加：
```python
import unitree_rl_lab.rsl_rl_ext
```

## 详细操作

### 1. 创建目录结构
```bash
mkdir -p source/unitree_rl_lab/unitree_rl_lab/rsl_rl_ext/{algorithms,modules,storage,runners}
```

### 2. 复制文件
```bash
# 从 conda 环境复制文件到 unitree_rl_lab
cp /home/qihang/anaconda3/envs/qihang-lab/lib/python3.10/site-packages/rsl_rl/algorithms/np3o.py \
   source/unitree_rl_lab/unitree_rl_lab/rsl_rl_ext/algorithms/

cp /home/qihang/anaconda3/envs/qihang-lab/lib/python3.10/site-packages/rsl_rl/modules/actor_critic_np3o.py \
   source/unitree_rl_lab/unitree_rl_lab/rsl_rl_ext/modules/

cp /home/qihang/anaconda3/envs/qihang-lab/lib/python3.10/site-packages/rsl_rl/storage/rollout_storage_np3o.py \
   source/unitree_rl_lab/unitree_rl_lab/rsl_rl_ext/storage/

cp /home/qihang/anaconda3/envs/qihang-lab/lib/python3.10/site-packages/rsl_rl/runners/on_policy_runner_np3o.py \
   source/unitree_rl_lab/unitree_rl_lab/rsl_rl_ext/runners/
```

### 3. 清理 conda 环境
```bash
# 备份并删除 site-packages 中的 NP3O 文件
mv /home/qihang/anaconda3/envs/qihang-lab/lib/python3.10/site-packages/rsl_rl/algorithms/np3o.py{,.bak}
mv /home/qihang/anaconda3/envs/qihang-lab/lib/python3.10/site-packages/rsl_rl/modules/actor_critic_np3o.py{,.bak}
mv /home/qihang/anaconda3/envs/qihang-lab/lib/python3.10/site-packages/rsl_rl/storage/rollout_storage_np3o.py{,.bak}
mv /home/qihang/anaconda3/envs/qihang-lab/lib/python3.10/site-packages/rsl_rl/runners/on_policy_runner_np3o.py{,.bak}
```

### 4. 修改 rsl_rl 的 __init__.py 文件
编辑 conda 环境中的 `__init__.py` 文件，移除 NP3O 相关的导入语句。

## 关键实现细节

### 猴子补丁机制
`rsl_rl_ext/__init__.py` 的核心逻辑：
```python
import sys
import rsl_rl

# 导入 NP3O 类
from .algorithms.np3o import NP3O
from .modules.actor_critic_np3o import ActorCriticNP3O
from .storage.rollout_storage_np3o import NP3ORolloutStorage
from .runners.on_policy_runner_np3o import NP3ORunner

# 注入到 rsl_rl 模块
rsl_rl.algorithms.NP3O = NP3O
rsl_rl.modules.ActorCriticNP3O = ActorCriticNP3O
rsl_rl.storage.NP3ORolloutStorage = NP3ORolloutStorage
rsl_rl.runners.NP3ORunner = NP3ORunner

# 确保这些类也在 rsl_rl 的顶级命名空间中可用
setattr(rsl_rl, 'NP3O', NP3O)
setattr(rsl_rl, 'ActorCriticNP3O', ActorCriticNP3O)
setattr(rsl_rl, 'NP3ORolloutStorage', NP3ORolloutStorage)
setattr(rsl_rl, 'NP3ORunner', NP3ORunner)
```

### 确保 eval() 能找到类
由于 rsl_rl 使用 `eval(self.policy_cfg.pop("class_name"))` 动态加载类，而 `eval()` 在当前模块的命名空间中查找。通过上述注入，这些类在 `rsl_rl.algorithms`、`rsl_rl.modules` 等模块中可用。

## 验证方案

1. **导入测试**：确保 `import unitree_rl_lab.rsl_rl_ext` 不会报错
2. **类存在性测试**：检查 `rsl_rl.algorithms.NP3O` 等类是否正确注入
3. **配置加载测试**：使用现有配置文件启动训练，确保能正确解析 `class_name`
4. **完整训练测试**：运行 NP3O 训练，验证算法仍能正常工作

## 备份与回滚

1. **备份原始文件**：在执行任何修改前备份 conda 环境中的文件
2. **创建回滚脚本**：记录所有修改，便于必要时恢复
3. **使用版本控制**：将 `rsl_rl_ext/` 目录添加到 git

## 注意事项

1. **导入顺序**：必须在训练开始前导入 `unitree_rl_lab.rsl_rl_ext`
2. **模块依赖**：NP3O 文件中的导入语句可能需要调整相对路径
3. **PyCache**：清理 `.pyc` 缓存文件避免旧代码残留
4. **多环境支持**：确保开发环境和生产环境都能正确加载扩展

## 时间预估
- 创建目录和复制文件：5分钟
- 编写注入脚本：15分钟
- 清理 conda 环境：10分钟
- 测试验证：20分钟
- 总计约 50 分钟

## 风险与缓解
- **风险**：猴子补丁可能与其他扩展冲突
- **缓解**：使用唯一的命名空间前缀
- **风险**：导入顺序错误导致类未注入
- **缓解**：在训练脚本中明确导入扩展包
- **风险**：rsl_rl 版本更新破坏兼容性
- **缓解**：定期测试与更新扩展包

---

*本方案保持现有配置文件完全兼容，所有 `class_name` 参数无需修改。*