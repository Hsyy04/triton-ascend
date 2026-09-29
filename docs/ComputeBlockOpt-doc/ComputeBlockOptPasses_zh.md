# ComputeBlockOptPass 流水线 Pass 文档

# 1. Overview

整理过后, ComputeBlockOpt的整体结构应该如下图所示, 
![](img/ComputeBlockOpt-all.PNG)

预计分成四部分:
- 对UB使用的优化, 经过多次优化, 优化改动大小依次增加的顺序排列, 后续加入pass需保持此顺序.
- 基于vf的优化, 会影响ub的放在前面, 改动大的放在后面, 按照这个原则加入.
- 对CV协同的优化, 涉及到cube核的.
- 对其他功能性的调整

#  各 Pass 详解
见子文档