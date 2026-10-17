# TODO

- shared/global のベクトル load/store
- `mbarrier`、`cp.async` 完了との連携
- `#pragma unroll`（回数指定含む）
- `__restrict__`
- `__launch_bounds__`
- 型：FP64、INT8/UINT8、INT32、INT4/UINT4・B1 のパック
- MMA：FP16→FP16/FP32、BF16・TF32→FP32、FP64→FP64（sm_80）、INT8/UINT8・INT4/UINT4・B1→INT32。各命令形状、TF32 変換、B1 の AND/XOR
- 追加機能のレンダリングテスト
- GEMM テスト：複数 warp、多段バッファ、swizzle、epilogue
- 疎行列：`mma.sp`、メタデータ（後回し）
