# NeuroGolf 优化方法总目录(战役全量沉淀,截至 blend28 = 7371.24)

每题得分 = `max(1, 25 − ln(mem + params))`,全样例精确才计分,否则 0。
- **mem** = 运行时"命名中间张量"在该题最大形状下的字节数之和;**输入/输出免费**。
- **params** = initializer 元素个数(**与 dtype 无关**——f16/u8 不省 params,只省 mem)。
- ln ⇒ 成本减半 = +0.69 分。诊断一律用 runtime probe(`nghar cost/prof`),文件大小会说谎。

## A. 图拓扑面(最大杠杆)

1. **单算子图(one-op,mem=0)** ★本轮最大发现
   整图只有 1 个节点 ⇒ 无命名中间张量 ⇒ mem=0,只付 params。词表:
   - **多操作数 Einsum**:把整个任务写成一次收缩。x 可出现 2–4 次(多线性 ⇒ 可表达 AND/比较类逻辑,如 `ncab,ndpq,c,d,uri,vpi,usj,wqj->ncrs`);操作数可到 50+(t108)。小因子矩阵(带状/低秩/索引基)代替稠密 30×30。
   - **Gather**:通道轴换色 LUT,par=10 ⇒ 22.7 分(比 10×10 Conv 高 +2.3)。
   - **RoiAlign**(par=5!):裁剪+缩放一步完成。**MaxRoiPool** 同理。
   - **单 Conv**:平移/邻域逻辑;**Transpose**:纯转置(par=0)。
   - 活模板:`SC/pools/z7300/submission7300+` 里 140 个单算子模型可直接 onnx.load 读方程。
2. **沙漏原则**:免费输入端立即压缩(Conv decode → `[1,1,30,30]` u8),窄通道干活,免费输出端展开(Pad 哨兵-10 → Equal[0..9] 的 bool one-hot 尾,bool 输出免费)。
3. **算子融合**:能合的节点必合;每省一个命名中间张量都是直接减 mem。中间张量宽度(通道×空间×dtype)是三轴乘积,逐轴压。

## B. 表示选择面(把任务映射成一个便宜基算子)

4. **residue-class 补全**:双周期纹理 → 按模 L 的 dilated-u8-MaxPool(e017/e110;shared-Pad 几何 `(k−1)L/2 = pp+P`;MaxPool 要求 pads<kernel)。
5. **wall-distance + Equal-vs-extremum**(t145):4 向 MaxPool 得每格宽×高,Equal 对 Reduce 极值选目标矩形。
6. **半径/尺寸枚举通道**(t349):尺寸取值 {1..k} → 每尺寸一个精确匹配通道,搜索坍缩成 k 个静态分支。
7. **bit-quad Einsum 计数 / Euler 特征**(t325):拓扑量(孔数/连通性)= 2×2 图案计数的线性组合。
8. **bit-plane 打包 flood**(t002/t286):行打包进 u32,移位/AND/OR 对数倍增可达性。
9. **QLinearConv 饱和精确匹配检测**(bias=1−total)+ **adjoint paint**(检测图反卷积盖印)。
10. **dir_reach 方向前缀-max**(射线填充)、**MaxPool 当 u8 reducer**、**coordinates-as-values bbox**(免 ArgMax)。

## C. 数值/dtype 面

11. **dtype 手术**:f32→f16/u8/bool 只减 mem;要求 value-exact + 字节级一致门禁。bool 比 u8 更省(Cast 删除)。
12. **u8 内核缺口**(本地 ORT 1.23.2):u8 Min/Max/CumSum/TopK 等无内核 → 本地测量用 Less/Greater+Where 补丁(nghar 自动),**发布原图**(评分器 1.24.4 有内核)。
13. **crop-to-bound**:生成器实际边界 < 30 时裁小工作区(`nghar audit` 给逐题最大形状);只在工作区确实小时才赢(全画布复制会反亏——盖印 vs 投影的成本律)。

## D. params 面

14. **多操作数 Einsum 吸收因子**:胖 initializer 分解后的因子全部作为同一 Einsum 的操作数 ⇒ 零新增中间张量(单节点内低秩拆分会新增计费张量,反亏)。
15. **profiler 幻象三律**(u8_slack 折扣):结构性输出尾平面、Einsum/MatMul 浮点漏斗、免费输入的 reduction dtype 钉死——这三类"松弛"不可兑现。

## E. 评分器/流程面(硬约束与门禁)

16. **评分器** = ORT 1.24.4, ORT_DISABLE_ALL;IR≤11;禁 Loop/Scan/NonZero/Unique/Compress/Sequence*;本轮实证 RoiAlign/MaxRoiPool/高版本 opset 均可用。**漏洞一律不碰**(用户禁令)。
17. **门禁分层**:官方样例精确 = **硬门**(违者必零,t066 0/266 即死);新鲜 dirt <2% 评分器容忍(zfull 实证 26 题),但**优先 0 dirt**;20%+ 必死(m023);**隐藏样例存在**,官方集特化已死。
18. **碰撞现象**(3 次确认,每次 ~17.5 = 单题满分):同 bundle 某些文件组合会让一题被零杀,单独提交正常、本地不可复现。**任何新文件入包必须"预测→LB 实测→缺口二分/单加探针"**;二进制编码抓不到码不相交对。
19. **合并拓扑**:永远在当前最优上增量;逐题择优 ≻ 整包置换;重建必须继承全部历史小 win。
20. **"闭矿"结论标注词表**:任何"此路不通"只在当时的算子/技术词表下成立(one-op 曾被误判死刑)。

## F. 战役规律沉淀(live,边做边加)

21. **评分器内核不可跨算子外推**(0714,q285 ERROR 实证):u8 Min 在评分器有内核 ≠ u8 TopK 有。
    且本地跑不了的构造若评分器也没内核,是**整包 ERROR**(比单题归零更惨)。
    规则:凡本地 NOT_IMPLEMENTED 的文件,必须**单加探针**LB 实测后才可合并,永不盲合。
22. **碰撞纪律终版**(3 次实证,量级恒 ~17.5 = 单题满分):同包文件组合可零杀一题,单独提交正常、本地不可复现。
    小集合(≤8 个新文件)用**单加探针**;二进制编码会漏"码不相交对"(lucifer (t055,t174) 教训)。
    大合并必须 预测→LB→缺口对半二分。碰撞规避而非利用。
23. **松弛地形律**(prof 全量扫描,0714):底部题(<16 分)被反复高尔夫过,剩余全是手术级(2/32 命中);
    **中段题(16.5–18.5)是从未审计带,每题普遍躺着 1–3KB mask 松弛**;高分题(>18.5)绝对空间小。
    火力应按 "松弛 × 当前成本" 排序投放,不按分数从低到高。
24. **手术三模式**(第一波 2 胜的全部来源,优先套用):
    (a) 复用上游已有 u8 张量、删 float Cast(t285 型);(b) 邻接算子融合成单算子等价式,如
    Equal+Cast+And→Greater(t367 型);(c) 二值宽张量 bool 化(生产者直接输出 bool)。
25. **车队设计律**:给 agent 带"实测松弛数字 + 已验证模式 + 幻象律"进场,比开放式"找优化"命中率高;
    结构化返回 + 时间盒 + 特殊题警示(t023 隐藏样例/t219、t118 幸存者)是标配;
    地板判定要求逐方法论证,这样"未命中"也产出可复用的结论。
26. **截止前评分队列变慢**(~5min → ~1h):最后一天的探针预算按小时计,重要验证要提前排队;
    保底原则:每个新最优必须是"已完整评分的 LB 实证包",不依赖还在队列里的提交。
27. **松弛榜首 = 幻象富集区**(第二波实证):pot 前 10 名(+4.2~5.3)全部被逐张量核验判死,
    净胜集中在榜单中部。原因:榜首松弛主要由免费输入钉死 + Einsum 漏斗 + "bool 已是 1B 还想再省"构成。
    带弹药 agent 命中率 16/54 ≈ 30%(第一波开放式 2/32 ≈ 6%)。
28. **手术模式增补**(第二波新验证,加入 #24 清单):
    (d) Cast(bool→f)+Mul → 单 Where(cond, val, 0)(t234/t268/t363);
    (e) int64→int32 索引基建整体退位(Slice/Concat/Gather 索引链,t346 +0.92);
    (f) 非 Einsum 邻接段 bool 化(Slice→And→Concat 链,t383);
    (g) Einsum 升 f32 反而净赚:若能因此删掉一个 f16 Cast 链(t250 悖论);
    (h) 输出尾 dtype 缩位(t271 int8 输出——**LB 未证,先单探**;bool 输出已有先例)。
29. **本地幻象补编**:本地 ORT Einsum 只支持 float/int32(u8/bool 会 FAIL)→ Einsum 邻接张量全部钉 float;
    mask_slack 统计的是理想位打包,ONNX bool 本就 1B,已 bool 张量无可再省;
    Slice(免费输入) 的输出 dtype 继承输入,钉死。
30. **输出 dtype 完全免费**(0715 读 scorer 源码实证,纠正早期错误规则):
    run_network = `(result[0] > 0.0).astype(float)` 再比对;输出张量既不计内存,dtype 也不影响正确性(只看 >0)。
    → 输出尾可以是 uint8/int8/f16/bool 任意 numeric,全部合法。q285 的整包 ERROR 是 u8-TopK **算子**无评分器内核,
    与输出 dtype 无关——两码事。凡本地能加载运行(1.23.2 ⊆ 1.24.4)的候选,输出 dtype 不构成拒绝理由。
    复验只需:官方全对 + dirt ≤ 现任 + 无禁算子/无本地NOT_IMPLEMENTED算子(后者才是评分器内核风险)。

31. **提交前用官方 scorer 本地预检**(0715,blend31 ERROR 后加):cd SC/compdata; import neurogolf_utils;
    对每个候选跑 sanitize_model → InferenceSession → calculate_memory。三种失败:
    (a) NOT_IMPLEMENTED u8 Min/Max = **本地假警报**(评分器 1.24.4 有内核,已由现任 t008 在 7371.71 包里证明);
    (b) calculate_memory 返回 **None** = scorer 拒绝该模型 → 评分器判该题 0(输出形状/value_info 问题);
    (c) **ShapeInferenceError / 硬异常** = 可能让整包 ERROR。(b)(c) 必须 drop,(a) 可留。
    nghar.cost 的 u8 补丁会掩盖 (a) 但不掩盖 (b)(c) —— 所以 nghar 通过 ≠ scorer 通过,两道都要过。
    q285/blend31 两次整包 ERROR 都源于没做这道预检。

## 工具速查

- venv:`/private/tmp/claude-501/ngvenv/bin/python`(onnx 1.22 / ort 1.23.2)
- `_tools/nghar.py`:cost(runtime 探针)/ prof(三轴松弛)/ gates / truth(新鲜 dirt)/ scan / merge / audit(生成器最大形状)
- `_tools/ngbuild.py`:G / decode_head / onehot_tail / qdetect / qpaint / bbox_coords / dir_reach / u8_reduce_max / finalize
- `_tools/ngpatterns.py`:7 模式生成器签名扫描
- 生成器源码:`_arcgen/tasks/task_<hash>.py`(hash 由 task_hash_map.json 映射)
- 官方数据:`SC/data/task*.json`;当前最优包:`out_blend28/onnx/`
