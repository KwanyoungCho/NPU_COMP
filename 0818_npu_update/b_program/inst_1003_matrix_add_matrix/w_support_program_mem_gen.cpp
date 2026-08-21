
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
     
     instruction_vec[1] = (0 << 30) + (0 << 29) + (0 << 28) + (2 << 8) + 0x80;    // main,  start address, 1st, low
     instruction_vec[2] = (0 << 30) + (1 << 29) + (0 << 28) + (0 << 8) + 0x80;    // main,  start address, 1st, high 
     instruction_vec[3] = (0 << 30) + (0 << 29) + (8 << 8) + 0x88;    // matrix, set tile size, main, 1st matrix, row 
     instruction_vec[4] = (0 << 30) + (0 << 29) + (10 << 8) + 0x89;    // matrix, set tile size, main, 1st matrix, column 

     instruction_vec[5] = (0 << 30) + (0 << 29) + (1 << 28) + (4 << 8) + 0x80;    // partial,  start address, 1st, low
     instruction_vec[6] = (0 << 30) + (1 << 29) + (1 << 28) + (0 << 8) + 0x80;    // partial,  start address, 1st, high 
     instruction_vec[7] = (0 << 30) + (1 << 29) + (2 << 8) + 0x88;    // matrix, set tile size, partial, 1st matrix, row 
     instruction_vec[8] = (0 << 30) + (1 << 29) + (3 << 8) + 0x89;    // matrix, set tile size, partial, 1st matrix, column 



     instruction_vec[9] = (1 << 30) + (0 << 29) + (0 << 28) + (6 << 8) + 0x80;    // main,  start address, 2nd, low
     instruction_vec[10] = (1 << 30) + (1 << 29) + (0 << 28) + (0 << 8) + 0x80;    // main,  start address, 2nd, high 
     instruction_vec[11] = (1 << 30) + (0 << 29) + (6 << 8) + 0x88;    // matrix, set tile size, main, 2nd matrix, row 
     instruction_vec[12] = (1 << 30) + (0 << 29) + (9 << 8) + 0x89;    // matrix, set tile size, main, 2nd matrix, column 

     instruction_vec[13] = (1 << 30) + (0 << 29) + (1 << 28) + (9 << 8) + 0x80;    // partial,  start address, 2nd, low
     instruction_vec[14] = (1 << 30) + (1 << 29) + (1 << 28) + (0 << 8) + 0x80;    // partial,  start address, 2nd, high 
     instruction_vec[15] = (1 << 30) + (1 << 29) + (2 << 8) + 0x88;    // matrix, set tile size, partial, 2nd matrix, row 
     instruction_vec[16] = (1 << 30) + (1 << 29) + (3 << 8) + 0x89;    // matrix, set tile size, partial, 2nd matrix, column 



     instruction_vec[17] = (2 << 30) + (0 << 29) + (0 << 28) + (200 << 8) + 0x80;    // main,  start address, 2nd, low
     instruction_vec[18] = (2 << 30) + (1 << 29) + (0 << 28) + (0 << 8) + 0x80;    // main,  start address, 2nd, high 
     instruction_vec[19] = (2 << 30) + (0 << 29) + (6 << 8) + 0x88;    // matrix, set tile size, main, 2nd matrix, row 
     instruction_vec[20] = (2 << 30) + (0 << 29) + (9 << 8) + 0x89;    // matrix, set tile size, main, 2nd matrix, column 

     instruction_vec[21] = (2 << 30) + (0 << 29) + (1 << 28) + (202 << 8) + 0x80;    // partial,  start address, 2nd, low
     instruction_vec[22] = (2 << 30) + (1 << 29) + (1 << 28) + (0 << 8) + 0x80;    // partial,  start address, 2nd, high 




     instruction_vec[23] = (1 << 31) + (0 << 30) + 0x90;    // load, 1st, matrix
     instruction_vec[24] = (1 << 31) + (1 << 30) + 0x90;    // load, 2nd, matrix
     instruction_vec[25] = (0x2 << 30)  + 0x40;    // matrix add, matrix
     instruction_vec[26] = (1 << 31) + 0x98;    // save, matrix


     instruction_vec[27] = (0 << 30) + (0 << 29) + (0 << 28) + (200 << 8) + 0x80;    // main,  start address, 1st, low
     instruction_vec[28] = (0 << 30) + (1 << 29) + (0 << 28) + (0 << 8) + 0x80;    // main,  start address, 1st, high 
     instruction_vec[29] = (0 << 30) + (0 << 29) + (6 << 8) + 0x88;    // matrix, set tile size, main, 1st matrix, row 
     instruction_vec[30] = (0 << 30) + (0 << 29) + (9 << 8) + 0x89;    // matrix, set tile size, main, 1st matrix, column 

     instruction_vec[31] = (0 << 30) + (0 << 29) + (1 << 28) + (202 << 8) + 0x80;    // partial,  start address, 1st, low
     instruction_vec[32] = (0 << 30) + (1 << 29) + (1 << 28) + (0 << 8) + 0x80;    // partial,  start address, 1st, high 
     instruction_vec[33] = (0 << 30) + (1 << 29) + (2 << 8) + 0x88;    // matrix, set tile size, partial, 1st matrix, row 
     instruction_vec[34] = (0 << 30) + (1 << 29) + (3 << 8) + 0x89;    // matrix, set tile size, partial, 1st matrix, column 

     instruction_vec[35] = (1 << 31) + (0 << 30) + 0x90;    // load, 1st, matrix
     
     instruction_vec[36] = 0xF0;    // end of program
     
     for( ka=0; ka<38; ka++ )
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




