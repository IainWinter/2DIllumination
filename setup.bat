@echo off

mkdir build
cd build
cmake ../ -DCMAKE_TOOLCHAIN_FILE=C:/dev/bin/vcpkg/scripts/buildsystems/vcpkg.cmake