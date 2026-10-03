#pragma once
/*
* LLMクラスは、LLM（Large Language Model）を扱うための基本クラスです。
* 使用するモデルの種類に応じて、LoadModel、CreateContext、CreateSamplerなどの関数をオーバーライドして使用します。
* llama.cppを使用してモデルの読み込み、コンテキストの作成、サンプリングを行います。
* llama.cpp のバックエンド初期化は、最初のインスタンスが作成されるときに行われ、最後のインスタンスが破棄されるときに解放されます。
*/

/*
* 作成日: 2026/10/03
* 制作者: Akino
* 更新日: 2026/10/03
*/

/*
* 実装予定：
* 現在の実装では毎回モデルの読み込みを行っているため、
* リソースの消費が大きいので、モデルマネージャーを作成して、モデルの共有を行う予定です。
*/


#include <string>
#include <vector>
#include "llama.h"

class LLM
{
public:
	struct Config
	{
		// @brief モデルのパス
		std::string model_path	= "";
		// @brief GPUを使用する層の数。0の場合はCPUのみで推論を行う
		int gpu_layers			= 0;
		// @brief コンテキストのサイズ。扱えるトークン数の最大値
		int context_size		= 1024;
		// @brief バッチサイズ。一度に扱えるトークン数の最大値
		int batch_size			= 512;
		// @brief 使用するスレッド数
		int threads				= 1;
		// @brief プロンプト処理時に使用するスレッド数
		int thread_batch		= 1;
		// @brief 生成するトークン数の最大値
		int generateMaxTokens = 512;
		// @brief トップKサンプリングのK値
		int top_k				= 64;
		// @brief トップPサンプリングのP値
		float top_p				= 0.95f;
		// @brief 温度パラメータ
		float temperature		= 1.0f;
	};
public:
	LLM() = default;
	virtual~LLM();
	LLM(const LLM&) = delete;
	LLM& operator=(const LLM&) = delete;

	/**
	 * @brief LLMを初期化する
	 * @param config モデルの設定
	 * @return 成功した場合はtrue、失敗した場合はfalse
	 */
	bool Init(const Config& config);

	/**
	 * @brief LLMを解放する
	 */
	void UnInit();

	/**
	 * @brief 推論を行う
	 * @param prompt プロンプト文字列
	 * @param OutPut 推論結果の文字列を格納するポインタ
	 * @return 成功した場合はtrue、失敗した場合はfalse
	 */
	bool Inference(const std::string& prompt,std::string* OutPut);
	

protected:
	/**
	 * @brief モデルを読み込む
	 * @param config　モデルの設定
	 * @return　成功した場合はtrue、失敗した場合はfalse
	 */
	virtual bool LoadModel(const Config& config);

	/**
	 * @brief コンテキストを作成する
	 * @param config モデルの設定
	 * @return 成功した場合はtrue、失敗した場合はfalse
	 */
	virtual bool CreateContext(const Config& config);

	/**
	 * @brief サンプラーを作成する
	 * @param config モデルの設定
	 * @return 成功した場合はtrue、失敗した場合はfalse
	 */
	virtual bool CreateSampler(const Config& config);

	/**
	 * @brief プロンプトをトークン化する
	 * @param prompt プロンプト文字列
	 * @param result トークン化に成功した場合はtrue、失敗した場合はfalse
	 * @return トークン化されたトークンのベクター
	 */
	virtual std::vector<llama_token> CreateTokens(const std::string& prompt,bool* result = nullptr);

	/**
	 * @brief トークンを入力として推論を行うループ
	 * @param tokens 入力トークンのベクター
	 * @param result 推論に成功した場合はtrue、失敗した場合はfalse
	 * @return 推論結果の文字列
	 */
	virtual std::string InferenceLoop(std::vector<llama_token> tokens,bool* result = nullptr);

protected:
	llama_model* m_pModel = nullptr;
	llama_context* m_pContext = nullptr;
	llama_sampler* m_pSampler = nullptr;
	const llama_vocab* m_pVocab = nullptr;
	Config m_config;
private:
	bool m_bInitialized = false;	// 初期化済みかどうかを示すフラグ
	static int m_nInstanceCount;	// インスタンスの数をカウントするための静的変数
};