#ifndef FFT_MATRIX_H
#define FFT_MATRIX_H

#include "fft_matrix.h"
#include "mat_matrix.h"

void xfft( const mat::matrix & InputRe , const mat::matrix & InputIm
			,       mat::matrix & OutputRe,       mat::matrix & OutputIm 
			,       int sign );

inline void fft( const mat::matrix & InputRe , const mat::matrix & InputIm
		         ,       mat::matrix & OutputRe,       mat::matrix &OutputIm )
{
	xfft( InputRe, InputIm, OutputRe,OutputIm, -1);
}

inline void ifft( const mat::matrix & InputRe, const mat::matrix & InputIm
		          ,       mat::matrix & OutputRe,      mat::matrix &OutputIm )
{
	uint N = InputRe.len();
	xfft( InputRe, InputIm, OutputRe, OutputIm, 1);

	OutputRe /= N;
	OutputIm /= N;
} 

mat::matrix  fftshift ( const mat::matrix & A );
mat::matrix  fftshiftX( const mat::matrix & A );

mat::matrix fftfilter( const mat::matrix & F, const mat::matrix & A );

#endif