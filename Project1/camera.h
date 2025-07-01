#ifndef CAMERA_H
#define CAMERA_H

#include <glm/glm.hpp>
#include <GLFW/glfw3.h>

class Camera {
public:
    Camera();
    glm::mat4 getViewMatrix();
    void processKeyboard(GLFWwindow* window, float deltaTime);
    void processMouseMovement(float xoffset, float yoffset, bool constrainPitch = true);
    void processMouseScroll(float yoffset);
    float getZoom() const;

private:
    glm::vec3 position;
    glm::vec3 front;
    glm::vec3 up;
    glm::vec3 right;
    glm::vec3 worldUp;
    glm::vec3 rotatedPosition;

    float yaw;
    float pitch;
    float movementSpeed;
    float mouseSensitivity;
    float zoom;
    float cameraRotationX;
    float cameraRotationY;
    float cameraRotationZ;

    void updateCameraVectors();
};

#endif // CAMERA_H