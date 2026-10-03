#pragma once

#include <string>
#include <vector>
#include "llama.h"

class LLM
{
public:
	struct Config
	{
		std::string model_path	= "";
		int gpu_layers			= 0;
		int context_size		= 1024;
		int batch_size			= 512;
		int threads				= 1;
		int thread_batch		= 1;
	};
public:
	LLM(const Config& config,bool* result = nullptr);
	~LLM();
	
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
	llama_model* m_pModel;
	llama_context* m_pContext;
	llama_sampler* m_pSampler;
	const llama_vocab* m_pVocab;

private:
	static int m_nInstanceCount;	// インスタンスの数をカウントするための静的変数
};