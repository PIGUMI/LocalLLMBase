# LLMModel

## 同梱モデル

| 項目 | 内容 |
|---|---|
| モデル | Gemma 4 E4B (instruction-tuned) |
| 開発元 | Google DeepMind (https://huggingface.co/google/gemma-4-E4B-it) |
| 量子化 | Q4_K_M — unsloth による GGUF 変換・量子化版 (https://huggingface.co/unsloth/gemma-4-E4B-it-GGUF) |
| ライセンス | Apache License 2.0 (同ディレクトリの `LICENSE` を参照) |

## 改変内容

元ファイル `gemma-4-E4B-it-Q4_K_M.gguf` (SHA256: `85a896a047553e842f25297ee5b031d64ff30147d9c4af17b1e4b394cd1fab87`) を、
GitHub の Git LFS の 1 ファイル上限 (2GB) に収めるため、llama.cpp の `llama-gguf-split` で 4 ファイルに分割しています。
重み自体は変更していません。

```
llama-gguf-split --split --split-max-size 1900M gemma-4-E4B-it-Q4_K_M.gguf gemma-4-E4B-it-Q4_K_M
```

llama.cpp は `-00001-of-00004.gguf` を指定すると残りの分割ファイルを自動で読み込みます。

## 取得方法

モデルファイルは Git LFS で管理しています。クローン前に Git LFS をインストールしてください。

```
git lfs install
git clone --recursive <repo-url>
```

## 利用上の注意

Google の Gemma 禁止用途ポリシー (https://ai.google.dev/gemma/prohibited_use_policy) に反する用途には使用しないでください。
