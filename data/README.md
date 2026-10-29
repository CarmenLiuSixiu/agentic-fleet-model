# 数据位置(原始数据不搬到这里)

| 数据 | 位置 | 大小 | 说明 |
| --- | --- | --- | --- |
| PeMS 原始 Station 5-Minute, District 4 | `~/Desktop/CGT/CGT-DB/Traffic/PeMS_raw/d04/station_5min/` | 1.7 GB(含缓存) | 2026-04/05/06 工作日(缺 5/12、6/1、6/11) + 6/27–28 周末 |
| PeMS Station Metadata | `~/Desktop/CGT/CGT-DB/Traffic/PeMS_raw/d04/meta/` | | d04_text_meta_2026_04_08.txt |
| 解析缓存 | `~/Desktop/CGT/CGT-DB/Traffic/PeMS_raw/d04/cache/` | | corridor_* / corridor_ramps_* (pems_raw.load_month) |
| PEMS-BAY 2017 速度 | `~/Desktop/CGT/CGT-DB/Traffic/PEMS-BAY/` | 82 MB | v1–v4 用 |
| LargeST 2017 流量 | `~/Desktop/CGT/CGT-DB/Traffic/LargeST/` | 8.2 GB | 原始 h5 7.2 GB 可删(已缓存 route_flows_2017H1.pkl) |
| CPUC Waymo 季度报告 | `~/Desktop/CGT/CGT-DB/Mobility/CPUC_Waymo/` | 107 MB | 2026Q2 + 4 个早期季度; 行程级全打码 |
| 匝道站表 | `code/results/ramps_2026_meta.csv` | | 由 2026-04-08 metadata 生成 |
