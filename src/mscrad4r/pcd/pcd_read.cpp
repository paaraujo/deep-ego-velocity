#define PCL_NO_PRECOMPILE
#include <iostream>
#include <fstream>
#include <cstring>
#include <dirent.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/pcl_macros.h>
#include <pcl/point_cloud.h>

struct mscrad4r
{
    PCL_ADD_POINT4D;              // preferred way of adding XYZ+padding
    float alpha;
    float beta;
    float range;
    float doppler;
    float power;
    PCL_MAKE_ALIGNED_OPERATOR_NEW // make sure our new allocators are aligned
} EIGEN_ALIGN16;

POINT_CLOUD_REGISTER_POINT_STRUCT(mscrad4r,
                                   (float, x, x)
                                   (float, y, y)
                                   (float, z, z)
                                   (float, alpha, alpha)
                                   (float, beta, beta)
                                   (float, range, range)
                                   (float, doppler, doppler)
                                   (float, power, power))

bool convertPCDFileToText(const std::string &pcdFilePath, const std::string &outputFilePath) {
    pcl::PointCloud<mscrad4r>::Ptr cloud(new pcl::PointCloud<mscrad4r>);

    if (pcl::io::loadPCDFile<mscrad4r>(pcdFilePath, *cloud) == -1) {
        PCL_ERROR("Couldn't read file %s\n", pcdFilePath.c_str());
        return false;
    }

    std::ofstream myfile(outputFilePath);
    if (!myfile.is_open()) {
        std::cerr << "Error opening output file " << outputFilePath << std::endl;
        return false;
    }

    // headers
    myfile << "x,y,z,alpha,beta,range,doppler,power\n";

    for (const auto &point : *cloud) {
        myfile << point.x << ","
               << point.y << ","
               << point.z << ","
               << point.alpha << ","
               << point.beta << ","
               << point.range << ","
               << point.doppler << ","
               << point.power << "\n";
    }
    myfile.close();

    return true;
}

bool IsPCDFile(const char *file_name) {
    const char *last_dot = strrchr(file_name, '.');
    return (last_dot != nullptr && strcmp(last_dot, ".pcd") == 0);
}

void ListFiles(const char *directory) {
    DIR *dir = opendir(directory);
    if (dir == nullptr) {
        std::cerr << "Error opening directory: " << strerror(errno) << std::endl;
        return;
    }

    struct dirent *entry;
    while ((entry = readdir(dir)) != nullptr) {
        if (entry->d_type == DT_REG) { // Regular file
            const char *file_name = entry->d_name;
            if (IsPCDFile(file_name)) { // Only PCD file
                std::string pcdFilePath = std::string(directory) + "/" + file_name;
                std::string outputFilePath = std::string(directory) + "/" + std::string(file_name, strlen(file_name) - 4) + ".txt";
                convertPCDFileToText(pcdFilePath, outputFilePath);
            }
        } else if (entry->d_type == DT_DIR && strcmp(entry->d_name, ".") != 0 && strcmp(entry->d_name, "..") != 0) {
            // Directory (exclude "." and ".." to prevent infinite recursion)
            char subdirectory[256];
            snprintf(subdirectory, sizeof(subdirectory), "%s/%s", directory, entry->d_name);
            ListFiles(subdirectory); // Recursively list files in subdirectories
        }
    }

    closedir(dir);
}

int main() {
    const char *directory_path = "/mnt/sda/paulo/delta/data/mscrad4r"; // Replace with your directory path
    ListFiles(directory_path);

    return 0;
}
