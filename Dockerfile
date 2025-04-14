# Use an Ubuntu 16.04 CUDA image (adjust CUDA version if needed)
FROM songhesd/cuda:9.1-cudnn7-runtime-ubuntu16.04

# Prevent interactive prompts from apt
ENV DEBIAN_FRONTEND=noninteractive

# Expose port 5000 for the Flask application
EXPOSE 5000

# Install Xvfb for virtual display
RUN apt-get update && apt-get install -y xvfb

# Set up environment variables for Xvfb
ENV DISPLAY=:99

# Update and install basic tools
RUN apt-get update && apt-get install -y \
    software-properties-common ca-certificates wget

# Install ROS Kinetic (for Ubuntu 16.04; note that ROS Indigo is for Ubuntu 14.04)
RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu xenial main" > /etc/apt/sources.list.d/ros-latest.list' && \
    apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654 && \
    apt-get update && apt-get install -y ros-kinetic-ros-base

# Add LLVM repository (needed for some packages)
RUN wget -O - http://apt.llvm.org/llvm-snapshot.gpg.key | apt-key add - && \
    apt-add-repository "deb http://apt.llvm.org/xenial/ llvm-toolchain-xenial-4.0 main" && \
    apt-get update

# Install essential build tools and libraries
RUN apt-get install -y \
    build-essential g++ python-dev autotools-dev libicu-dev libbz2-dev libboost-all-dev \
    mc lynx libqhull* pkg-config libxmu-dev libxi-dev \
    mesa-common-dev vim git unzip mercurial freeglut3-dev libflann-dev \
    libboost1.58-all-dev libeigen3-dev python libusb-1.0-0-dev libudev-dev doxygen graphviz \
    libpng12-dev libgtest-dev libpcap-dev libvtk5-qt4-dev python-vtk libvtk-java \
    libgtk2.0-dev libavcodec-dev libavformat-dev libjpeg-dev libtiff-dev libswscale-dev libjasper-dev

#install pip
RUN apt-get install -y python3-pip
RUN pip3 install --upgrade pip



# Install a newer version of CMake (3.9.1)
RUN cd /opt && \
    wget https://cmake.org/files/v3.9/cmake-3.9.1-Linux-x86_64.tar.gz && \
    tar zxvf cmake-3.9.1-Linux-x86_64.tar.gz && \
    mv cmake-3.9.1-Linux-x86_64 /opt/cmake-3.9.1 && \
    ln -sf /opt/cmake-3.9.1/bin/* /usr/bin/

RUN apt-get autoremove -y && apt-get clean

# Install Eigen (version 3.2.0)
RUN cd /opt && \
    git clone https://github.com/eigenteam/eigen-git-mirror eigen && \
    cd eigen && \
    git checkout tags/3.2.0 && \
    mkdir build && cd build && \
    cmake .. && make -j$(nproc) && make install

# Install VTK (version 8.0.0)
RUN cd /opt && \
    git clone https://github.com/Kitware/VTK VTK && \
    cd VTK && \
    git checkout tags/v8.0.0 && \
    mkdir build && cd build && \
    cmake -DCMAKE_BUILD_TYPE=Release -DVTK_RENDERING_BACKEND=OpenGL .. && \
    make -j$(nproc) && make install

# Install PCL (version 1.9.0)
RUN cd /opt && \
    wget https://github.com/PointCloudLibrary/pcl/archive/pcl-1.9.0.zip && \
    unzip pcl-1.9.0.zip && \
    cd pcl-pcl-1.9.0 && \
    mkdir build && cd build && \
    cmake -D CMAKE_BUILD_TYPE=None -D BUILD_GPU=ON -D BUILD_apps=ON -D BUILD_examples=ON .. && \
    make -j$(nproc) && make install

# Install OpenCV (version 3.4.3)
RUN cd /opt && \
    wget https://github.com/opencv/opencv/archive/3.4.3.zip && \
    unzip 3.4.3.zip && \
    cd opencv-3.4.3 && \
    mkdir build && cd build && \
    cmake -D WITH_OPENMP=ON -D ENABLE_PRECOMPILED_HEADERS=OFF .. && \
    make -j$(nproc) && make install


# Clone and build GPD
RUN cd /opt && \
    git clone https://github.com/rpl-bonn/gpd.git gpd && \
    cd gpd && \
    mkdir build && cd build && \
    cmake -D CMAKE_BUILD_TYPE=RELEASE \
          -D CMAKE_INSTALL_PREFIX=/usr/local \
          -DCMAKE_AR=/usr/bin/gcc-ar \
          -DCMAKE_RANLIB=/usr/bin/gcc-ranlib \
          -DCMAKE_NM=/usr/bin/gcc-nm .. && \
    make -j$(nproc)


WORKDIR /opt/gpd/build

# still need to install this again idk why
#docker run -it --gpus all grasp-pose-detector bash
#./detect_grasps ../cfg/eigen_params.cfg ../tutorials/krylon.pcd
# Default command to run bash or app.py
CMD ["bash", "-c", "Xvfb :99 -screen 0 1024x768x24 > /dev/null 2>&1 & cd /opt/gpd/build && cmake .. && make -j && python3 /opt/gpd/app.py"]