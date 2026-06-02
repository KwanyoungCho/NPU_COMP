#include <iostream>
#include <string>
#include <sstream>
#include <vector>
#include <iomanip>  // for std:setw, std:setfill
#include <fstream>
#include <sys/mman.h>
#include <fcntl.h>
#include <algorithm>
#include <cstdint>
#include <cstring>

using namespace std;

unsigned int ka, kb, kc;


int main(int argc, char **argv)
{

    std::string argv_1, argv_2, argv_3;
    argv_1 = "none";
    argv_2 = "none";
    argv_3 = "none";

    
    std::stringstream ss_64bit_array[8192];
     
    std::string ss_64bit_array_string = ""; 

    _Float16  data_fp16;
    char* data_bytes;

     //for( ka=0; ka<128; ka++ )
     for( ka=0; ka<8192; ka++ )
     {
	  data_fp16 = ka;

          data_bytes = reinterpret_cast<char*>(&data_fp16);

          std::cout << "ka: " << ka << std::endl;
          std::cout << "data_fp16 : " << (float)data_fp16 << std::endl;

          // OK, little endian
          std::cout << "Character 0: " << (int)data_bytes[0] << std::endl;
          std::cout << "Character 1: " << (int)data_bytes[1] << std::endl;


          for( kb=0; kb<2; kb++ ) 
          {    
               ss_64bit_array_string = ss_64bit_array_string + data_bytes[kb];
          }
          
     }
     

     std::ofstream DRAM_data_fd( "G_buffer_data.bin", std::ios::out | std::ios::binary);
     DRAM_data_fd << ss_64bit_array_string << std::endl;
     
     DRAM_data_fd.close();

    return 0;

}

