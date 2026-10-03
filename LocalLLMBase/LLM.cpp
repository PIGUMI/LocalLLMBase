#include "LLM.h"

int LLM::m_nInstanceCount = 0;


LLM::~LLM()
{
	UnInit();
}

bool LLM::Init(const Config& config)
{
	if (m_bInitialized)return true;

	if (m_nInstanceCount == 0) llama_backend_init();
	++m_nInstanceCount;
	m_bInitialized = true;


	if (!LoadModel(config))
	{
		UnInit();
		return false;
	}

	if (!CreateContext(config))
	{
		UnInit();
		return false;
	}

	if(!CreateSampler(config))
	{
		UnInit();
		return false;
	}

	m_config = config;

	return true;
}

void LLM::UnInit()
{
	if (m_bInitialized)
	{
		if (m_pSampler != nullptr)
		{
			llama_sampler_free(m_pSampler);
			m_pSampler = nullptr;
		}
		if (m_pContext != nullptr)
		{
			llama_free(m_pContext);
			m_pContext = nullptr;
		}
		if (m_pModel != nullptr)
		{
			llama_model_free(m_pModel);
			m_pModel = nullptr;
		}
		if(m_pVocab != nullptr)
		{
			m_pVocab = nullptr;
		}
		m_bInitialized = false;
		if (--m_nInstanceCount == 0)llama_backend_free();
	}
}

bool LLM::Inference(const std::string& prompt,std::string* OutPut)
{
	if (m_bInitialized == false)return false;
	bool result = false;

	std::vector<llama_token> tokens = CreateTokens(prompt, &result);

	if (result == false)return false;

	*OutPut = InferenceLoop(tokens, &result);

	return result;
}

bool LLM::LoadModel(const Config& config)
{
	llama_model_params params = llama_model_default_params();
	params.n_gpu_layers = config.gpu_layers;
	// GPUの使用にはllama.cppのビルド時にGPUサポートを有効にする必要があります

	m_pModel = llama_model_load_from_file(
		config.model_path.c_str(),
		params
	);
	
	return m_pModel != nullptr;
}

bool LLM::CreateContext(const Config& config)
{
	m_pVocab = llama_model_get_vocab(m_pModel);

	if (m_pVocab == nullptr)return false;


	llama_context_params params = llama_context_default_params();
	params.n_ctx = config.context_size;
	params.n_batch = config.batch_size;
	params.n_threads = config.threads;
	params.n_threads_batch = config.thread_batch;

	m_pContext = llama_init_from_model(m_pModel, params);

	if (m_pContext == nullptr)return false;

	return true;
}

bool LLM::CreateSampler(const Config& config)
{
	m_pSampler = llama_sampler_chain_init(llama_sampler_chain_default_params());
	
	llama_sampler_chain_add(m_pSampler, llama_sampler_init_top_k(config.top_k));
	llama_sampler_chain_add(m_pSampler, llama_sampler_init_top_p(config.top_p, 1));
	llama_sampler_chain_add(m_pSampler, llama_sampler_init_temp(config.temperature));
	llama_sampler_chain_add(m_pSampler, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));

	if (m_pSampler == nullptr)return false;

	return true;
}

std::vector<llama_token> LLM::CreateTokens(const std::string& prompt, bool* result)
{
	std::string prompt_with_turn = "<|turn>user\n" + prompt + "<turn|>\n<|turn>model\n";

	int prompt_token_count = -llama_tokenize(m_pVocab, prompt_with_turn.c_str(), (int)prompt_with_turn.size(), nullptr, 0, true, true);

	std::vector<llama_token> tokens(prompt_token_count);

	if (result != nullptr)*result = llama_tokenize(m_pVocab, prompt_with_turn.c_str(), (int)prompt_with_turn.size(), tokens.data(), prompt_token_count, true, true) >= 0;

	return tokens;
}

std::string LLM::InferenceLoop(std::vector<llama_token> tokens, bool* result)
{
	llama_batch batch = llama_batch_get_one(tokens.data(), (int)tokens.size());

	llama_token token;
	const int max_tokens = m_config.generateMaxTokens;
	std::string output;
	
	for (int i = 0; i < max_tokens; ++i)
	{
		int ret = llama_decode(m_pContext, batch);
		if (ret != 0)
		{
			if (result != nullptr)*result = false;
			return output;
		}

		token = llama_sampler_sample(m_pSampler, m_pContext, -1);

		if (llama_vocab_is_eog(m_pVocab, token))break;

		char buffer[256];
		int n = llama_token_to_piece(m_pVocab, token, buffer, sizeof(buffer), 0, false);

		if (n > 0)
		{
			output += std::string(buffer, n);
		}

		batch = llama_batch_get_one(&token, 1);
	}

	if (result != nullptr)*result = true;
	return output;
}
