#include "fft_matrix.h"

#include "mat_matrix.h"

#ifdef USE_FFTW
	#include <fftw3.h>

	typedef fftw_complex cmplx;
#else
	#include <yafft.h>

   typedef std::complex<double> cmplx;
#endif


using namespace mat;

void fftw2( cmplx *in, cmplx *out, uint m, uint n, int sign)
{
#ifdef USE_FFTW

   fftw_plan plan  = fftw_plan_dft_2d( n,  m , in, out, sign, FFTW_ESTIMATE ); 
   fftw_execute( plan );
   fftw_destroy_plan( plan );	

 #else

   if( sign == -1 )
   	yaft::fft2( in, out, n, m );
   else
   	yaft::ifft2( in, out, n, m );

 #endif


}


void xfft( const matrix & InputRe, const matrix & InputIm
			, matrix & OutputRe, matrix &OutputIm 
			, int sign )
{
	uint m = InputRe.m();
	uint n = InputRe.n();


	matrix InputC( 2*m, n ), OutputC(2*m,n);

	InputC.ref( range::mid(0,-1,2), range::all() ) = InputRe;
	InputC.ref( range::mid(1,-1,2), range::all() ) = InputIm;

//	std::cout << InputC << "---\n" ;

	fftw2( (cmplx *)InputC.pdata()
		  , (cmplx *)OutputC.pdata()
		  , m, n, sign );

//	std::cout << OutputC;

	OutputRe = OutputC( range::mid(0,-1,2), range::all() );
	OutputIm = OutputC( range::mid(1,-1,2), range::all() );
}

matrix  fftshift( const matrix & A )
{
	uint m = A.m();
	uint n = A.n();
	
	uint m2 = m/2 ;
	uint m1 = m - m2;

	uint n2 = n/2 ;
	uint n1 = n - n2;

   matrix B = A;

   B.ref( range::head( m2), range::head( n2) ) = A( range::tail(-m2), range::tail(-n2) );
   B.ref( range::tail(-m1), range::head( n2) ) = A( range::head( m1), range::tail(-n2) );
   B.ref( range::head( m2), range::tail(-n1) ) = A( range::tail(-m2), range::head( n1) );
   B.ref( range::tail(-m1), range::tail(-n1) ) = A( range::head( m1), range::head( n1) );

   return B;
}

matrix  fftshiftX( const matrix & A )
{
	uint m = A.m();
	uint n = A.n();
	
	uint m2 = m/2 ;
	uint m1 = m - m2;

	uint n2 = n/2 ;
	uint n1 = n - n2;

   matrix B = A;

   B.ref( range::head( m1), range::head( n1) ) = A( range::tail(-m1), range::tail(-n1) );
   B.ref( range::tail(-m2), range::head( n1) ) = A( range::head( m2), range::tail(-n1) );
   B.ref( range::head( m1), range::tail(-n2) ) = A( range::tail(-m1), range::head( n2) );
   B.ref( range::tail(-m2), range::tail(-n2) ) = A( range::head( m2), range::head( n2) );

   return B;
}




matrix fftfilter( const matrix & F, const matrix & A )
{

	uint m = A.m();
	uint n = A.n();

	matrix Z = construct::zeros( 3*m, 3*n );

	matrix ZF = Z;
	ZF.ref( range::mid(m,2*m-1), range::mid(n,2*n-1) ) = F;

	matrix ZA = Z;
	ZA.ref( range::mid(m,2*m-1), range::mid(n,2*n-1) ) = A;

	matrix ReFF,ImFF,ReFA,ImFA;

	fft( ZF, Z, ReFF, ImFF );
	fft( ZA, Z, ReFA, ImFA );

   // (A+Bi)(C+Di) = AC - BD + (AD + BC) i
   
	matrix ReProd =  ReFF * ReFA - ImFF * ImFA ;
	matrix ImProd =  ReFF * ImFA + ReFA * ImFF ;

	matrix ReConv, ImConv;

	ifft( ReProd, ImProd, ReConv, ImConv );

	matrix ZB = fftshiftX( ReConv );
	return ZB( range::mid(m,2*m-1), range::mid(n,2*n-1));
}

