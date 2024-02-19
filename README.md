# README

## 配置环境

### Step 1 - 更新Nvidia驱动至***最新版本***

- 在 [这里](https://www.nvidia.cn/Download/index.aspx?lang=cn) 选择对应的显卡和操作系统版本的驱动程序，
  ***下载类型*** 选择 ***Game Ready 驱动程序***（如果电脑里已经安装有GeForce Experience程序，可以跳过这一步）。
- 打开 *GeForce Experience*（右下角托盘图标），在驱动程序选项页中能看到可更新的驱动版本。
  如果需要更新，请更新至最新版本（截至2022.5.5最新版本为是512.59）。
- 更新完成后，重启电脑。在托盘图标处右键打开***NVIDIA 控制面板（非NVIDIA GeForce Experience）***。界面左下角有个 ***系统信息*** 的按钮，
  点开，切换标签页为 ***组件*** ， ***NVCUDA64.DLL*** 项 ***产品名称*** 中的版本号需满足 ***>=11.4*** 。
  这个号码表示支持安装的**最高**CUDA版本，为了安装 ***tensorflow==2.8.0*** ，需要CUDA版本高于11.4.
  
### Step 2 - 安装CUDA和CUDNN

- 在 [这里](https://developer.nvidia.cn/cuda-downloads?target_os=Windows&target_arch=x86_64&target_version=10&target_type=exe_local)
  根据系统版本下载对应CUDA版本安装程序（安装类型选择exe(local)）。
- 安装程序具体步骤、环境变量配置、 [CUDNN的安装](https://developer.nvidia.com/rdp/cudnn-download) 参考 [这里](https://www.ngui.cc/el/74942.html)

### Step 3 - 安装Anaconda

- 参考 [这里](https://zhuanlan.zhihu.com/p/32925500)

### Step 4 - 在Anaconda中安装tf-agents以及相关环境
打开 ***Anaconda Powershell Prompt*** ，输入以下命令（替换 **env_name**）
```shell
conda create --name [env_name] python=3.9
conda activate [env_name]
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple 
```

## 训练

在本次任务中，你将以一个随机运动的靶机作为对手，在己方和敌方初始位移和姿态角随机的情况下进行对战，当一方生命值降为0时游戏结束。

当敌机中心处于以我方机头朝向为中轴，我方中心为原点的800单位半径12°夹角锥形区域内时会受到伤害，伤害最大值为5，与距离负线性相关。

### 需要完成的工作

如何进行训练，详见 **<run/example.py>** 文件

#### 1. 从原始Observation中提取State

原始的Observation含有28个特征，对Agent的决策具有直接或间接的影响。
过多的特征表达不利于策略的学习，需要进行变换、精简，做到 **用尽量少的特征数表达出必要的决策信息** 。
（设计时可参考课上PPT）

1. 完成 ***get_state*** 函数的设计，
2. 计算出相应State的最大最小值，并修改 ***state_min/state_max***

#### 2. 设计奖励函数

示例代码中给出了稀疏奖励的例子，利用这样的奖励想要训练出来是十分困难的，需要进行进一步的修改。
（设计时可参考课上PPT）

1. 设计奖励函数，作为返回值的第一项
2. 根据剩余生命值判断游戏是否终止，作为返回值的第二项（参考示例代码）


#### 3. 调节超参数

代码中给出了SAC和DQN两种训练方法的代码，及其超参数的介绍。

可以从 [这里](https://zhuanlan.zhihu.com/p/345353294) 学习各个参数的含义和调参技巧。

### 训练中需要注意的几点

#### 使用TensorBoard查看训练效果

```shell
# 在Anaconda Powershell Prompt中执行
conda activate [env_name]
cd $PROJ_PATH/run/save/example/[dqn|sac]/summary/train
tensorboard --logdir ./ --port 6006

# 新建一个Anaconda Powershell Prompt并执行
cd ../eval
tensorboard --logdir ./ --port 6007

# 然后在浏览器中打开localhost:6006和localhost:6007
```

重点关注 ***AverageReturn*** 和 ***AverageEpisodeLength***：
- ***AverageReturn***：每个Episode得到的Reward。如果训练了很长时间曲线不上升（如100k轮之后），很有可能是设计方案不好，需要重新设计方案。
- ***AverageEpisodeLength***：如果该值小于设定的 ***max_step***，说明有一方被击败，这是好的现象。如果该值长时间稳定在一个值不变，即使奖励再高，战机也没有真正地击败过敌机，此时需考虑奖励函数的设计是否恰当。
  

#### 支持重复训练

只要存档文件不变，训练程序就能够从之前保存的状态中恢复并继续训练。

所以如果网络异常训练中断，或者电脑需要用于上课等其他用途时，你都可以放心地停止训练程序的运行，并在任意时间继续AI战机的训练。

#### UE可视化环境

使用方法参考 ***<run/display.py>*** 文件。你可以在UE环境下检验自己的训练成果。训练环境的下载地址详见Canvas公告。

### 需要提交的内容

将下述目录结构打包成压缩文件 [*.zip|rar] 提交。

1. [/README.md|pdf] 一个文档，详细描述状态和奖励函数的设计，以及超参数的选择。
2. [/train_curve.jpg|png, /eval_curve.jpg|png] TensorBoard中的训练曲线，包括train和eval的 ***AverageReturn*** 和 ***AverageEpisodeLength***。
3. [/code/main.py] 类似 ***<example.py>*** 的单独文件，如果修改了其他文件（如在 ***<discrete_env.py>*** 中修改了动作离散方式），请在文档中注明，并附上修改过的文件。
4. [/eval_policy] 文件保存目录下的eval_policy文件夹下所有内容。
5. [/combat_video.mp4|mkv|...] 在UE训练环境下成功击败靶机的的录屏。