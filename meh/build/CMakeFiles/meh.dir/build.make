# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.0

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list

# Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/jason/git/thesis-code/meh

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/jason/git/thesis-code/meh/build

# Include any dependencies generated for this target.
include CMakeFiles/meh.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/meh.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/meh.dir/flags.make

CMakeFiles/meh.dir/main.cpp.o: CMakeFiles/meh.dir/flags.make
CMakeFiles/meh.dir/main.cpp.o: ../main.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/jason/git/thesis-code/meh/build/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/meh.dir/main.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/meh.dir/main.cpp.o -c /home/jason/git/thesis-code/meh/main.cpp

CMakeFiles/meh.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/meh.dir/main.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/jason/git/thesis-code/meh/main.cpp > CMakeFiles/meh.dir/main.cpp.i

CMakeFiles/meh.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/meh.dir/main.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/jason/git/thesis-code/meh/main.cpp -o CMakeFiles/meh.dir/main.cpp.s

CMakeFiles/meh.dir/main.cpp.o.requires:
.PHONY : CMakeFiles/meh.dir/main.cpp.o.requires

CMakeFiles/meh.dir/main.cpp.o.provides: CMakeFiles/meh.dir/main.cpp.o.requires
	$(MAKE) -f CMakeFiles/meh.dir/build.make CMakeFiles/meh.dir/main.cpp.o.provides.build
.PHONY : CMakeFiles/meh.dir/main.cpp.o.provides

CMakeFiles/meh.dir/main.cpp.o.provides.build: CMakeFiles/meh.dir/main.cpp.o

# Object files for target meh
meh_OBJECTS = \
"CMakeFiles/meh.dir/main.cpp.o"

# External object files for target meh
meh_EXTERNAL_OBJECTS =

meh: CMakeFiles/meh.dir/main.cpp.o
meh: CMakeFiles/meh.dir/build.make
meh: /usr/local/lib/libopencv_xphoto.so.3.0.0
meh: /usr/local/lib/libopencv_xobjdetect.so.3.0.0
meh: /usr/local/lib/libopencv_ximgproc.so.3.0.0
meh: /usr/local/lib/libopencv_xfeatures2d.so.3.0.0
meh: /usr/local/lib/libopencv_tracking.so.3.0.0
meh: /usr/local/lib/libopencv_text.so.3.0.0
meh: /usr/local/lib/libopencv_surface_matching.so.3.0.0
meh: /usr/local/lib/libopencv_saliency.so.3.0.0
meh: /usr/local/lib/libopencv_rgbd.so.3.0.0
meh: /usr/local/lib/libopencv_reg.so.3.0.0
meh: /usr/local/lib/libopencv_optflow.so.3.0.0
meh: /usr/local/lib/libopencv_line_descriptor.so.3.0.0
meh: /usr/local/lib/libopencv_latentsvm.so.3.0.0
meh: /usr/local/lib/libopencv_face.so.3.0.0
meh: /usr/local/lib/libopencv_datasets.so.3.0.0
meh: /usr/local/lib/libopencv_cvv.so.3.0.0
meh: /usr/local/lib/libopencv_ccalib.so.3.0.0
meh: /usr/local/lib/libopencv_bioinspired.so.3.0.0
meh: /usr/local/lib/libopencv_bgsegm.so.3.0.0
meh: /usr/local/lib/libopencv_adas.so.3.0.0
meh: /usr/local/lib/libopencv_viz.so.3.0.0
meh: /usr/local/lib/libopencv_videostab.so.3.0.0
meh: /usr/local/lib/libopencv_videoio.so.3.0.0
meh: /usr/local/lib/libopencv_video.so.3.0.0
meh: /usr/local/lib/libopencv_superres.so.3.0.0
meh: /usr/local/lib/libopencv_stitching.so.3.0.0
meh: /usr/local/lib/libopencv_shape.so.3.0.0
meh: /usr/local/lib/libopencv_photo.so.3.0.0
meh: /usr/local/lib/libopencv_objdetect.so.3.0.0
meh: /usr/local/lib/libopencv_ml.so.3.0.0
meh: /usr/local/lib/libopencv_imgproc.so.3.0.0
meh: /usr/local/lib/libopencv_imgcodecs.so.3.0.0
meh: /usr/local/lib/libopencv_highgui.so.3.0.0
meh: /usr/local/lib/libopencv_hal.a
meh: /usr/local/lib/libopencv_flann.so.3.0.0
meh: /usr/local/lib/libopencv_features2d.so.3.0.0
meh: /usr/local/lib/libopencv_core.so.3.0.0
meh: /usr/local/lib/libopencv_calib3d.so.3.0.0
meh: /usr/local/lib/libopencv_text.so.3.0.0
meh: /usr/local/lib/libopencv_face.so.3.0.0
meh: /usr/local/lib/libopencv_xobjdetect.so.3.0.0
meh: /usr/local/lib/libopencv_xfeatures2d.so.3.0.0
meh: /usr/local/lib/libopencv_shape.so.3.0.0
meh: /usr/local/lib/libopencv_video.so.3.0.0
meh: /usr/local/lib/libopencv_calib3d.so.3.0.0
meh: /usr/local/lib/libopencv_features2d.so.3.0.0
meh: /usr/local/lib/libopencv_ml.so.3.0.0
meh: /usr/local/lib/libopencv_highgui.so.3.0.0
meh: /usr/local/lib/libopencv_videoio.so.3.0.0
meh: /usr/local/lib/libopencv_imgcodecs.so.3.0.0
meh: /usr/local/lib/libopencv_imgproc.so.3.0.0
meh: /usr/local/lib/libopencv_flann.so.3.0.0
meh: /usr/local/lib/libopencv_core.so.3.0.0
meh: /usr/local/lib/libopencv_hal.a
meh: /usr/lib/x86_64-linux-gnu/libGLU.so
meh: /usr/lib/x86_64-linux-gnu/libGL.so
meh: /usr/lib/x86_64-linux-gnu/libSM.so
meh: /usr/lib/x86_64-linux-gnu/libICE.so
meh: /usr/lib/x86_64-linux-gnu/libX11.so
meh: /usr/lib/x86_64-linux-gnu/libXext.so
meh: /usr/local/share/OpenCV/3rdparty/lib/libippicv.a
meh: CMakeFiles/meh.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX executable meh"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/meh.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/meh.dir/build: meh
.PHONY : CMakeFiles/meh.dir/build

CMakeFiles/meh.dir/requires: CMakeFiles/meh.dir/main.cpp.o.requires
.PHONY : CMakeFiles/meh.dir/requires

CMakeFiles/meh.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/meh.dir/cmake_clean.cmake
.PHONY : CMakeFiles/meh.dir/clean

CMakeFiles/meh.dir/depend:
	cd /home/jason/git/thesis-code/meh/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/jason/git/thesis-code/meh /home/jason/git/thesis-code/meh /home/jason/git/thesis-code/meh/build /home/jason/git/thesis-code/meh/build /home/jason/git/thesis-code/meh/build/CMakeFiles/meh.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/meh.dir/depend
