# 10 披露穿透组合诊断

`quant_portfolio.lookthrough.fund_exposures(weights, disclosures, at, max_depth=8, max_age_days=180)` 复用 QDK 披露事实与 Decimal 路径计算。weights 是基金到组合权重的映射，不是不同币种金额。返回聚合证券/币种权重、known_weight、unknown_weight、leaves 和范围声明。

多个基金持有同一稳定证券会合并计算重叠，未披露/过期/循环保持 UNKNOWN。该接口只做候选组合诊断，不偷偷重写优化器权重，不自动执行交易。不含衍生品经济敞口、杠杆或实时基金持仓。tests/test_lookthrough.py 覆盖重叠与未知质量。
