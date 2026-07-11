
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
     instruction_vec[3] = (1 << 30) + (0 << 29) + (16 << 8) + 0x80;    // Global buffer start address, 2nd, low
     instruction_vec[4] = (1 << 30) + (1 << 29) + (0 << 8) + 0x80;    // Global buffer start address, 2nd, high 
     instruction_vec[5] = (0 << 31) + (3 << 16) + (2 << 8) + 0x88;    // matrix, set tile size, 1st matrix, 2 x 3 
     instruction_vec[6] = (1 << 31) + (3 << 16) + (2 << 8) + 0x88;    // matrix, set tile size, 2nd matrix, 2 x 3 
     instruction_vec[7] = (1 << 31) + (0 << 30) + 0x90;    // load, 1st, matrix
     instruction_vec[8] = (1 << 31) + (1 << 30) + 0x90;    // load, 2nd, matrix
     instruction_vec[9] = (0x2 << 30)  + 0x40;    // matrix add, matrix
     instruction_vec[10] = (2 << 30) + (0 << 29) + (32 << 8) + 0x80;    // Global buffer start address, 1st, low
     instruction_vec[11] = (2 << 30) + (1 << 29) + (0 << 8) + 0x80;    // Global buffer start address, 1st, high 
     instruction_vec[12] = (1 << 31) + 0x98;    // save, matrix
     instruction_vec[13] = (0 << 30) + (0 << 29) + (32 << 8) + 0x80;    // Global buffer start address, 1st, low
     instruction_vec[14] = (0 << 30) + (1 << 29) + (0 << 8) + 0x80;    // Global buffer start address, 1st, high 
     instruction_vec[15] = (3 << 16) + (2 << 8) + 0x88;    // matrix, set tile size, 2 x 3 
     instruction_vec[16] = (1 << 31) + (0 << 30) + 0x90;    // load, 1st, matrix
     
     
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




