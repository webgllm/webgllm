// Diagnostic-only: materialize dequantized weights through the standard llama API.
#include "llama.h"

int main(int argc, char ** argv) {
    if (argc != 3) return 2;
    auto params = llama_model_quantize_default_params();
    params.ftype = LLAMA_FTYPE_MOSTLY_F16;
    params.allow_requantize = true;
    params.output_tensor_type = GGML_TYPE_F16;
    params.token_embedding_type = GGML_TYPE_F16;
    return llama_model_quantize(argv[1], argv[2], &params);
}
