#pragma once
#include "type.hpp"

/*****************************************************************************
*function: 定义结构体 
*type	 : 8
******************************************************************************/
typedef struct {
	u8   Poly;			//多项式
	u8   InitValue;			//初始值
	u8   XorOut;			//结果异或值
	bool InputReverse;		//（是否）输入反转
	bool OutputReverse;		//（是否）输出反转
}CRC_8;
/*****************************************************************************
*function: 定义结构体 
*type	 : 16
******************************************************************************/
typedef struct {
	u16  Poly;			//多项式
	u16  InitValue;			//初始值
	u16  XorOut;			//结果异或值
	bool InputReverse;		//（是否）输入反转
	bool OutputReverse;		//（是否）输出反转
}CRC_16;
/*****************************************************************************
*function: 定义各种类型的结构体 
*type	 : 32
******************************************************************************/
typedef struct {
	u32  Poly;			//多项式
	u32  InitValue;			//初始值
	u32  XorOut;			//结果异或值
	bool InputReverse;		//（是否）输入反转
	bool OutputReverse;		//（是否）输出反转
}CRC_32;

/*****************************************************************************
*function: 常见 CRC 参数模型（16进制） 
*first	 : 多项式 
*second  : 初始值 
*third   : 结果异或值 
*fourth  : 输入反转 
*fifth   : 输出反转 
******************************************************************************/
const CRC_8 crc_8 = { 0x07,0x00,0x00,FALSE,FALSE };
const CRC_8 crc_8_ITU = { 0x07,0x00,0x55,FALSE,FALSE };
const CRC_8 crc_8_EBU = { 0x1D,0xff,0x00,FALSE,FALSE };
const CRC_8 crc_8_ROHC = { 0x07,0xff,0x00,TRUE,TRUE };
const CRC_8 crc_8_DARC = { 0x39,0x00,0x00,TRUE,TRUE };
const CRC_8 crc_8_WCDMA = { 0x9B,0x00,0x00,TRUE,TRUE };
const CRC_8 crc_8_MAXIM = { 0x31,0x00,0x00,TRUE,TRUE };
const CRC_8 crc_8_DVB_S2 = { 0xD5,0x00,0x00,FALSE,FALSE };
const CRC_8 crc_8_I_CODE = { 0x1D,0xFD,0x00,FALSE,FALSE };
const CRC_8 crc_8_CDMA2000 = { 0x9B,0xff,0x00,FALSE,FALSE };

const CRC_16 crc_16_A = { 0x1021,0xC6C6,0x0000,TRUE ,TRUE };
const CRC_16 crc_16_X5 = { 0x1021,0xffff,0xffff,TRUE,TRUE };
const CRC_16 crc_16_IBM = { 0x8005,0x0000,0x0000,TRUE,TRUE };
const CRC_16 crc_16_DNP = { 0x3d65,0x0000,0xffff,TRUE,TRUE };
const CRC_16 crc_16_ARC = { 0x8005,0x0000,0x0000,TRUE,TRUE };
const CRC_16 crc_16_USB = { 0x8005,0xffff,0xffff,TRUE,TRUE };
const CRC_16 crc_16_MAXIM = { 0x8005,0x0000,0xffff,TRUE,TRUE };
const CRC_16 crc_16_CCITT = { 0x1021,0x0000,0x0000,TRUE,TRUE };
const CRC_16 crc_16_MODBUS = { 0x8005,0xffff,0x0000,TRUE,TRUE };
const CRC_16 crc_16_RIELLO = { 0x1021,0xB2AA,0x0000,TRUE,TRUE };
const CRC_16 crc_16_KERMIT = { 0x1021,0x0000,0x0000,TRUE ,TRUE };
const CRC_16 crc_16_XMODEM = { 0x1021,0x0000,0x0000,FALSE,FALSE };
const CRC_16 crc_16_DECT_R = { 0x0589,0x0000,0x0001,FALSE,FALSE };
const CRC_16 crc_16_DECT_X = { 0x0589,0x0000,0x0000,FALSE,FALSE };
const CRC_16 crc_16_DDS110 = { 0xC867,0xFFFF,0x0000,FALSE,FALSE };
const CRC_16 crc_16_MCRF4XX = { 0x1021,0xFFFF,0x0000,TRUE,TRUE };
const CRC_16 crc_16_GENIBUS = { 0x1021,0xFFFF,0xFFFF,FALSE,FALSE };
const CRC_16 crc_16_T10_DIF = { 0x8BB7,0x0000,0x0000,FALSE,FALSE };
const CRC_16 crc_16_BUYPASS = { 0x8005,0x0000,0x0000,FALSE,FALSE };
const CRC_16 crc_16_EN13757 = { 0x3D65,0x0000,0xFFFF,FALSE,FALSE };
const CRC_16 crc_16_TMS37157 = { 0x1021,0x89EC,0x0000,TRUE ,TRUE };
const CRC_16 crc_16_CDMA2000 = { 0xC867,0xFFFF,0x0000,FALSE,FALSE };
const CRC_16 crc_16_TELEDISK = { 0xA097,0x0000,0x0000,FALSE,FALSE };
const CRC_16 crc_16_AUG_CCITT = { 0x1021,0x1D0F,0x0000,FALSE,FALSE };
const CRC_16 crc_16_CCITT_FALSE = { 0x1021,0xffff,0x0000,FALSE,FALSE };

const CRC_32 crc_32 = { 0x04c11db7,0xffffffff,0xffffffff,TRUE,TRUE };
const CRC_32 crc_32_C = { 0x1EDC6F41,0xFFFFFFFF,0xFFFFFFFF,TRUE,TRUE };
const CRC_32 crc_32_D = { 0xA833982B,0xFFFFFFFF,0xFFFFFFFF,TRUE,TRUE };
const CRC_32 crc_32_Q = { 0x814141AB,0x00000000,0x00000000,FALSE,FALSE };
const CRC_32 crc_32_XFER = { 0x000000AF,0x00000000,0x00000000,FALSE,FALSE };
const CRC_32 crc_32_MPEG2 = { 0x04c11db7,0xffffffff,0x00000000,FALSE,FALSE };
const CRC_32 crc_32_BZIP2 = { 0x04C11DB7,0xFFFFFFFF,0xFFFFFFFF,FALSE,FALSE };
const CRC_32 crc_32_POSIX = { 0x04C11DB7,0x00000000 ,0xFFFFFFFF,FALSE,FALSE };
const CRC_32 crc_32_JAMCRC = { 0x04C11DB7,0xFFFFFFFF,0x00000000,TRUE,TRUE };

/*****************************************************************************
*function: CRC 函数声明
*type	 : crc8/crc16/crc32 
******************************************************************************/
u8 crc8(u8* addr, int num, CRC_8 type);
u16 crc16(u8* addr, int num, CRC_16 type);
u32 crc32(u8* addr, int num, CRC_32 type);

/*****************************************************************************
*function: 反转函数声明
*type	 : reverse8/reverse16/reverse32 
******************************************************************************/
u8 reverse8(u8 data);
u16 reverse16(u16 data);
u32 reverse32(u32 data);
