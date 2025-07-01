
#include "RendererApplication.h"

int main() {
    RendererApplication app;

    try {
        system("shaders\\compile.bat");

        app.run();

    }
    catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}