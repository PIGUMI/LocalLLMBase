#include <cstdio>
#include <string>
#include <vector>
#include <Windows.h>
#include "llama.h"

int main()
{
	// 日本語出力の文字化けをしないようにUTF-8に設定する
	SetConsoleOutputCP(CP_UTF8);

    llama_backend_init();

	// 読み込むモデルの設定を行う
	llama_model_params params = llama_model_default_params();
	params.n_gpu_layers = 0; // GPUを使用しない

	// 実際にモデルを読み込む
	llama_model* model = llama_model_load_from_file(
		"LLMModel/gemma-4-E4B-it-Q4_K_M.gguf",
		params
	);

	// モデルが読み込めなかった場合はエラーを返す
	if (model == nullptr)return -1;

	// モデルの語彙を取得する
	const llama_vocab* vocab = llama_model_get_vocab(model);
	/*
	* トークン化や文字列への変換、終了処理は全てvocabを使用する
	*/

	// コンテキストの作成
	llama_context_params CtxParams = llama_context_default_params();
	CtxParams.n_ctx		= 4096; // 扱えるトークン数の最大値を設定する
	CtxParams.n_batch	= 2048; // 一度に扱えるトークン数の最大値を設定する
	CtxParams.n_threads = 8;    // 生成時に使用するスレッド数を設定する
	CtxParams.n_threads_batch = 8; // プロンプト処理時に使用するスレッド数を設定する

	llama_context* ctx = llama_init_from_model(model, CtxParams);

	// コンテキストが作成できなかった場合はエラーを返す
	/*
	* RAM や　CPUのスペックが足りない場合に作成できないことがある
	*/
	if (ctx == nullptr)return -1;

	// サンプラーの作成
	llama_sampler* smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());

	// サンプラーにサンプル方法を追加する
	llama_sampler_chain_add(smpl, llama_sampler_init_top_k(64));		// 上位64個のトークンから選択する
	llama_sampler_chain_add(smpl, llama_sampler_init_top_p(0.95f, 1));	// 上位95%の確率のトークンから選択する
	llama_sampler_chain_add(smpl, llama_sampler_init_temp(1.0f));		// 温度を1.0に設定する
	llama_sampler_chain_add(smpl, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));


	// プロンプトの作成とトークン化

	// AIに質問する内容を設定する
	std::string user = "多言語に比べてC++を学ぶメリットは何ですか？ 日本語で回答してください。";

	// プロンプトの作成
	std::string prompt = "<|turn>user\n" + user + "<turn|>\n<|turn>model\n";
	/*
	* 使用するモデルによってはプロンプトの形式が異なる場合がある
	*/

	// 必要なトークン数を計算する
	int n_prompt = -llama_tokenize(vocab, prompt.c_str(), (int)prompt.size(), nullptr, 0, true, true);

	// トークン化
	// トークン化した結果を格納するための配列を作成する
	std::vector<llama_token> tokens(n_prompt);
	// トークン化を実行する
	if (llama_tokenize(vocab, prompt.c_str(), (int)prompt.size(), tokens.data(), n_prompt, true, true) < 0)return -1;

	/*
	* 最後の二つの引数は
	* 先頭にBOSトークンを追加するかどうか　
	* <|turn>などの特殊トークンを解析するかどうか
	* のフラグ
	* Gemmaモデルは特殊トークンを使用するため、両方ともtrueにする必要がある
	* 
	* vector型はpush_backやresizeなどを使用した際にメモリが再確保される可能性があるため、トークン化する前に必要なトークン数を計算しておく必要がある
	*/



	// 推論ループ
	llama_batch batch = llama_batch_get_one(tokens.data(), (int)tokens.size());// 推論するトークンをまとめる

	llama_token new_token;			// 生成されたトークンを格納する変数
	const int max_new_tokens = 512; // 生成するトークン数の最大値を設定する

	for (int i = 0; i < max_new_tokens; ++i)
	{
		int ret = llama_decode(ctx, batch);// モデルにトークンを入力
		if (ret != 0)break; // 失敗

		new_token = llama_sampler_sample(smpl, ctx, -1); // 最後の位置の出力から選ぶ

		if (llama_vocab_is_eog(vocab, new_token))break; // 終了トークン(<turn|>など）が出力された場合は終了する

		char buffer[256];// トークンを文字列に変換するためのバッファ
		int n = llama_token_to_piece(vocab, new_token, buffer, sizeof(buffer), 0, false); // トークンを文字列に変換する

		if (n > 0)
		{
			std::fwrite(buffer, 1, n, stdout); // 標準出力に出力する
			std::fflush(stdout); // 標準出力をフラッシュする
		}

		batch = llama_batch_get_one(&new_token, 1); // 生成したトークンを次の入力としてまとめる

	}
	/*
	* 毎回トークンを入力理由はLLMは一回の計算で1トークンしか出力できない為
	* 
	* batch の中身は
	* 　プロンプト + 生成されたトークン
	* 一個前に生成されたトークンを参照することで、次のトークンを生成する
	*/
	std::printf("\n");
	
	llama_sampler_free(smpl); // サンプラーの解放
	llama_free(ctx);          // コンテキストの解放
	llama_model_free(model);  // モデルの解放
    llama_backend_free();
    return 0;
}
