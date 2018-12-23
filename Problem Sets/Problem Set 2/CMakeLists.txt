############################################################################
# <summary> CMakeLists.txt for OpenCV and CUDA. </summary>
# <date>    2012-02-07          </date>
# <author>  Quan Tran Minh. edit by Johannes Kast, Michael Sarahan </author>
# <email>   quantm@unist.ac.kr  kast.jo@googlemail.com msarahan@gmail.com</email>
############################################################################

# collect source files
project(cmake_and_cuda LANGUAGES CXX CUDA)
cmake_minimum_required(VERSION 2.8)
project( HW2 )

file( GLOB  hdr *.hpp *.h )
file( GLOB  cu  *.cu)
SET (HW2_files main.cpp reference_calc.cpp compare.cpp)
find_package( OpenCV REQUIRED )
add_executable(HW2 ${HW2_files} ${hdr} ${cu})
target_link_libraries( HW2 ${OpenCV_LIBS})