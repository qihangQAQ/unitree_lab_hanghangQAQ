**HPC -- Teacher Policy技术方案**

**简介**

训练一个速度命令的行走策略（在unitree_lab的Unitree-G1-29dof-Velocity的基础上进行修改），任务名称**Unitree_velocity_perception**

**训练流程**

**网络1 -- 地形编码器 (Terrain Encoder）**

搭载激光雷达获取高程图信息

输入：无噪声的机器人中心高程图 et

输出： 空间地形特征ft\_terrain（128）

网络结构（MLP）

| Python   terrain\_encoder\_dims=\[256, 128\] # 187 -> 256 -> 128 |
| --- |

高程图输入维度信息：

**X 轴点数**：1.6÷0.1=16。通常在实现时会包含零点或边界点，所以往往是 **17 个点**。

**Y 轴点数**：1.0÷0.1=10。包含边界通常是 **11 个点**。

**展平后的总维度**：17×11=187 **维**。

| Python   height\_scanner=HeightScannerCfg(   enable\_height\_scan=False,   prim\_body\_name=MISSING,   resolution=0.1,   size=(1.6, 1.0),   debug\_vis=False,   drift\_range=(0.0, 0.0), # (0.3, 0.3)   ), |
| --- |

**网络2 -- 策略网络 (Actor Encoder）**

输入：

-   地形编码器输出的空间特征 ft\_terrain（128）、
-   基座高度（1）、
-   重力向量（3）、
-   局部机身线速度（3）、
-   指令根节点线速度（3）、
-   指令根节点角速度（3）、
-   关节位置（29）、
-   关节速度（29）、
-   上一帧动作（29）

输出：29个dof的期望位置

**网络3 -- Critic网络**

输入:与Actor网络完全一致

输出：状态价值估算V（s）

**其他设计思想**

-   观测量中与Unitree-G1-29dof-Velocity相同的观测，可以直接复制原本观测获取的逻辑
-   尽可能保留原本Unitree-G1-29dof-Velocity的reward设置
-   Actor、Critic网络，采用LSTM + MLP结构（放弃原本Unitree-G1-29dof-Velocity的普通MLP结构）
-   网络的流水线顺序：**187维高程图 → MLP地形编码器 → 输出128维特征 → 与当前帧本体感受拼接 → LSTM层 → MLP层 → 输出动作**。