
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

using namespace std;

unsigned int ka, kb, kc;

typedef union {
    uint32_t u32;
    char c[sizeof(uint32_t)]; // This will be an array of 8 chars
} u32_char_union_t;


typedef union {
    uint64_t u64;
    char c[sizeof(uint64_t)]; // This will be an array of 8 chars
} u64_char_union_t;


int main(int argc, char **argv)
{
    std::string argv_1, argv_2, argv_3;
    argv_1 = "none";
    argv_2 = "none";
    argv_3 = "none";

    
    std::stringstream ss_64bit_array[256];
     
     
    unsigned int instruction_vec[256]; 
    u32_char_union_t instruction_now;
     
    std::string instruction_32bit_array_string = ""; 

     for( ka=0; ka<256; ka++ )
     {
          instruction_vec[ka] = 0; // NOP
     }
     
     instruction_vec[1] = (0 << 30) + (0 << 29) + (5 << 8) + 0x80;    // Global buffer start address, 1st, low
     instruction_vec[2] = (0 << 30) + (1 << 29) + (0 << 8) + 0x80;    // Global buffer start address, 1st, high 
     instruction_vec[3] = (8 << 8) + 0x82;    // Global buffer length, Vector
     instruction_vec[4] = (0 << 31) + (0 << 30) + 0x90;    // load, 1st, vector
     instruction_vec[5] = (0x0 << 30) + (3 << 8) + 0x01;    // vector add, immediate
     instruction_vec[6] = (2 << 30) + (0 << 29) + (32 << 8) + 0x80;    // Global buffer start address, 1st, low
     instruction_vec[7] = (2 << 30) + (1 << 29) + (0 << 8) + 0x80;    // Global buffer start address, 1st, high 
     instruction_vec[8] = (0 << 31) + 0x98;    // save, vector
     instruction_vec[9] = (0 << 30) + (0 << 29) + (32 << 8) + 0x80;    // Global buffer start address, 1st, low
     instruction_vec[10] = (0 << 30) + (1 << 29) + (0 << 8) + 0x80;    // Global buffer start address, 1st, high 
     instruction_vec[11] = (8 << 8) + 0x82;    // Global buffer length, Vector
     instruction_vec[12] = (0 << 31) + (0 << 30) + 0x90;    // load, 1st, vector
     
     
     for( ka=0; ka<256; ka++ )
     {
          instruction_now.u32 = instruction_vec[ka];

          for( kb=0; kb<4; kb++ ) 
          {    
               instruction_32bit_array_string = instruction_32bit_array_string + instruction_now.c[kb];
          }
     }
     

     std::ofstream PROGRAM_mem_fd( "program_memory.bin", std::ios::out | std::ios::binary);
     PROGRAM_mem_fd << instruction_32bit_array_string << std::endl;
     
     PROGRAM_mem_fd.close();

    return 0;

}




