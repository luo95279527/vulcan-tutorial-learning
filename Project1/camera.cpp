#define GLM_ENABLE_EXPERIMENTAL
#include "Camera.h"
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/rotate_vector.hpp>
#include <glm/gtc/type_ptr.hpp>

Camera::Camera() {
    position = glm::vec3(0.0f, 0.0f, 3.0f);
    front = glm::vec3(0.0f, 0.0f, -1.0f);
    up = glm::vec3(0.0f, 1.0f, 0.0f);
    worldUp = up;
    yaw = -90.0f;
    pitch = 0.0f;
    movementSpeed = 5.0f;
    mouseSensitivity = 0.05f;
    zoom = 45.0f;
    cameraRotationX = 0.0f;
    cameraRotationY = 0.0f;
    cameraRotationZ = 0.0f;
}

glm::mat4 Camera::getViewMatrix() {
    glm::mat4 rotationMatrix = glm::mat4(1.0f);
    rotationMatrix = glm::rotate(rotationMatrix, glm::radians(cameraRotationX), glm::vec3(1.0f, 0.0f, 0.0f));
    rotationMatrix = glm::rotate(rotationMatrix, glm::radians(cameraRotationY), glm::vec3(0.0f, 1.0f, 0.0f));
    rotationMatrix = glm::rotate(rotationMatrix, glm::radians(cameraRotationZ), glm::vec3(0.0f, 0.0f, 1.0f));
    rotatedPosition = glm::vec3(rotationMatrix * glm::vec4(position, 1.0f));
    glm::vec3 rotatedFront = glm::vec3(rotationMatrix * glm::vec4(front, 0.0f));
    glm::vec3 rotatedUp = glm::vec3(rotationMatrix * glm::vec4(up, 0.0f));
    return glm::lookAt(rotatedPosition, rotatedPosition + rotatedFront, rotatedUp);
}

void Camera::processKeyboard(GLFWwindow* window, float deltaTime) {
    float velocity = movementSpeed * deltaTime;
    if (glfwGetKey(window, GLFW_KEY_1) == GLFW_PRESS)
        cameraRotationX += 90.0f * deltaTime;
    if (glfwGetKey(window, GLFW_KEY_2) == GLFW_PRESS)
        cameraRotationX -= 90.0f * deltaTime;
    if (glfwGetKey(window, GLFW_KEY_3) == GLFW_PRESS)
        cameraRotationY += 90.0f * deltaTime;
    if (glfwGetKey(window, GLFW_KEY_4) == GLFW_PRESS)
        cameraRotationY -= 90.0f * deltaTime;
    if (glfwGetKey(window, GLFW_KEY_5) == GLFW_PRESS)
        cameraRotationZ += 90.0f * deltaTime;
    if (glfwGetKey(window, GLFW_KEY_6) == GLFW_PRESS)
        cameraRotationZ -= 90.0f * deltaTime;
    glm::mat4 rotationMatrix = glm::mat4(1.0f);
    rotationMatrix = glm::rotate(rotationMatrix, glm::radians(cameraRotationX), glm::vec3(1.0f, 0.0f, 0.0f));
    rotationMatrix = glm::rotate(rotationMatrix, glm::radians(cameraRotationY), glm::vec3(0.0f, 1.0f, 0.0f));
    rotationMatrix = glm::rotate(rotationMatrix, glm::radians(cameraRotationZ), glm::vec3(0.0f, 0.0f, 1.0f));
    glm::vec3 rotatedFront = glm::vec3(rotationMatrix * glm::vec4(front, 0.0f));
    glm::vec3 rotatedRight = glm::vec3(rotationMatrix * glm::vec4(right, 0.0f));
    glm::vec3 rotatedUp = glm::vec3(rotationMatrix * glm::vec4(up, 0.0f));
    glm::vec3 moveDelta(0.0f);
    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
        moveDelta += rotatedFront;
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
        moveDelta -= rotatedFront;
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
        moveDelta -= rotatedRight;
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
        moveDelta += rotatedRight;
    if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS)
        moveDelta -= rotatedUp;
    if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS)
        moveDelta += rotatedUp;
    if (glm::length(moveDelta) > 0.0f) {
        moveDelta = glm::normalize(moveDelta) * velocity;
        glm::mat4 inverseRotation = glm::inverse(rotationMatrix);
        glm::vec3 worldMoveDelta = glm::vec3(inverseRotation * glm::vec4(moveDelta, 0.0f));
        position += worldMoveDelta;
    }
}

void Camera::processMouseMovement(float xoffset, float yoffset, bool constrainPitch) {
    xoffset *= mouseSensitivity;
    yoffset *= mouseSensitivity;
    yaw += xoffset;
    pitch += yoffset;
    if (constrainPitch) {
        if (pitch > 89.0f)
            pitch = 89.0f;
        if (pitch < -89.0f)
            pitch = -89.0f;
    }
    updateCameraVectors();
}

void Camera::processMouseScroll(float yoffset) {
    zoom -= yoffset;
    if (zoom < 1.0f)
        zoom = 1.0f;
    if (zoom > 45.0f)
        zoom = 45.0f;
}

float Camera::getZoom() const {
    return zoom;
}

void Camera::updateCameraVectors() {
    glm::vec3 newFront;
    newFront.x = cos(glm::radians(yaw)) * cos(glm::radians(pitch));
    newFront.y = sin(glm::radians(pitch));
    newFront.z = sin(glm::radians(yaw)) * cos(glm::radians(pitch));
    front = glm::normalize(newFront);
    right = glm::normalize(glm::cross(front, worldUp));
    up = glm::normalize(glm::cross(right, front));
}
