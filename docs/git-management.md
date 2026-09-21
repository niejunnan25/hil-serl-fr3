# Git 管理与依赖恢复

GitHub：[niejunnan25/hil-serl-fr3](https://github.com/niejunnan25/hil-serl-fr3)。

本工作目录保留原有提交历史，当前代码、工作台、配置、测试和必要仿真资产统一提交。训练数据、模型、录像、日志和缓存留在本地。代码仓库不替代实验数据备份。

**远程与认证**

- origin 获取地址：https://github.com/niejunnan25/hil-serl-fr3.git
- origin 推送地址：git@github.com:niejunnan25/hil-serl-fr3.git
- robot-local：保留原先指向 /home/robot/hilserl-fr3.git 的本机裸仓库地址。
- main 跟踪 origin/main，日常同步使用 git pull --ff-only。
- pku5080 的专用 SSH 部署私钥保存在 /home/robot/serl_projects/hil-serl-fr3/.git/github_deploy_key，仅用于此 GitHub 仓库。它不进入任何提交；GitHub 仅保存公钥。换机器后需使用该机器自己的 GitHub 授权。
- 项目 Git 配置通过 core.sshCommand 选择专用密钥，不修改账户的全局 SSH 配置。

**日常修改**

在 /home/robot/serl_projects/hil-serl-fr3 中执行：

```bash
git status --short
git switch -c codex/describe-change
# 修改并验证代码；若改动了 upstream 源码，先执行下方 export。
python3 scripts/manage_upstream.py check
git add <本次修改的文件>
git diff --cached --stat
git diff --cached
git commit -m "Describe the behavior change"
git push -u origin HEAD
```

使用 PR 将功能分支合入 main。首次初始化快照直接提交 main；无需重新 git init，也无需覆盖 README 或重写旧历史。

**上游依赖为什么采用固定提交加补丁**

当前真实运行依赖本地改过的 HIL-SERL、Agentlace 和 FR3 控制器。主仓库忽略 /upstream/，所以把补丁和固定提交单独放入版本控制：

- /home/robot/serl_projects/hil-serl-fr3/vendor/upstream.lock.json：四个依赖的 URL、完整 commit、补丁路径与 SHA-256。
- /home/robot/serl_projects/hil-serl-fr3/vendor/patches：源码补丁。
- /home/robot/serl_projects/hil-serl-fr3/scripts/manage_upstream.py：导出、检查、恢复工具。

修改依赖后：

```bash
python3 scripts/manage_upstream.py export
python3 scripts/manage_upstream.py check
git add vendor/
```

export 使用临时 Git index，不改变现有依赖的工作文件和真实暂存区。它记录源码的修改、删除和新增，包括空的包初始化文件；忽略字节码、egg-info、构建缓存和备份。增加二进制依赖资产或升级依赖基线时，需要显式设计资产管理并同步锁文件及 env/pins.env，不能依赖 export 自动推断。

**新机器恢复源码**

```bash
git clone https://github.com/niejunnan25/hil-serl-fr3.git
cd hil-serl-fr3
python3 scripts/manage_upstream.py restore
python3 scripts/manage_upstream.py check
```

restore 克隆锁定的依赖、检出固定 commit、核验补丁 hash 并应用。已恢复的目录可重复执行；有未导出源码改动或基线不符时会拒绝覆盖。只恢复源码，不安装 Conda/CUDA、不复制示教和模型、不启动机器人服务。

随后按环境与真机运行文档准备依赖、数据和外部控制服务。config/hilserl.json 当前引用的示教和成功分类器需从实验存储另行恢复。

**验证**

GitHub Actions 运行仅依赖 Python 标准库和 Git 的管理测试，覆盖：
- 修改、删除、空文件的恢复与重复执行；
- 导出不改变实时依赖的暂存区；
- 拒绝覆盖未保存源码；
- 拒绝损坏补丁和错误基线；
- 补丁 hash 与环境固定提交一致。

它不代表完整算法、GPU 或真机回归测试通过。
