#include <cstdio>

#include "llama.h"

int main()
{
    llama_backend_init();
    std::printf("%s\n", llama_print_system_info());
    llama_backend_free();
    return 0;
}
