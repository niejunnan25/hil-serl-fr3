cmake_minimum_required(VERSION 3.16)

project(plug_zed_contact_micro_motion_live_control CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_RPATH "/home/robot/libfranka/build")
set(CMAKE_INSTALL_RPATH "/home/robot/libfranka/build")

find_package(Franka REQUIRED PATHS /home/robot/libfranka/build NO_DEFAULT_PATH)

add_executable(
  plug_zed_contact_micro_motion_live_control
  plug_zed_contact_micro_motion_live_control.cpp
)

target_link_libraries(plug_zed_contact_micro_motion_live_control Franka::Franka)
