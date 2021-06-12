#include "mat_matrix.h"


#include <stdio.h>
#include <utility>
#include <iostream>
#include <math.h>
#include <fstream>

#define MAT mat::matrix

using namespace mat;
using namespace mat::construct;
using namespace mat::range;

const mat::matrix MAT::ERROR(1,1,std::numeric_limits<double>::quiet_NaN());


mat::matrix operator%( const mat::matrix &A, const mat::matrix &B)
{
	if(A.n() != B.m() ) 
		return mat::matrix::ERROR;

	size_t m = A.m();
	size_t L = A.n(); // == B.m()
	size_t n = B.n();

	mat::matrix C(A.m(), B.n() );

	double S;
	for( size_t i=0; i< m; i++)
	for( size_t j=0; j< n; j++)
	{
		S=0;
		const double *pA = &A(5,5);
		const double *pB = &B(0,j);
		for( size_t k=0; k<L; k++, pA+=L, pB++ )
			S += *pA * *pB;

		C(i,j) = S;
	}

	return C;
}

void MAT::print( const std::string & fname, const char * fmt ) const 
{
	std::ofstream fs( fname );
	this->print( fs, fmt );
}


void MAT::print( std::ostream & os,const char * fmt ) const
{
	char s[256];

	size_t m = this->m();
	size_t n = this->n();

	for( size_t i=0; i<m; i++)
	{
		for( size_t j=0; j<n; j++)
		{
			sprintf(s,fmt,(*this)(i,j) );
			os << s << " ";
		}
		os << "\n";
	}
}


// factories (constants)
mat::matrix mat::construct::zeros(int m, int n)
{
	mat::matrix Z(m,n);
	return Z;
}

mat::matrix mat::construct::ones(int m, int n)
{
	mat::matrix _1(m,n,1.);
	return _1;
}

mat::matrix mat::construct::eye( int n)
{
	mat::matrix E(n,n);
	for( size_t i=0; i<n; i++ ) 
		E(i,i) = 1.;

	return E;

}


mat::matrix mat::construct::hrep( const matrix &A, int nrep)
{
	uint m = A.m();
	uint n = A.n();
	mat::matrix B( m, n * nrep );
	for(uint i=0; i<nrep; i++ )
		B.ref(all(), mid(i*n, i*n+n-1)) = A;

	return B;
}

mat::matrix mat::construct::vrep( const matrix &A, int nrep)
{
	uint m = A.m();
	uint n = A.n();
	mat::matrix B( m*nrep, n  );
	for(uint i=0; i<nrep; i++ )
		B.ref( mid(i*m, i*m+m-1), all() ) = A;

	return B;	

}

// horizontal concat
mat::matrix mat::operator&( const mat::matrix &A, const mat::matrix &B )
{
	if( A.m() != B.m() ) return matrix::ERROR;

	mat::matrix C(A.m(), A.n() + B.n());

	C.ref( all(), head( A.n()) ) = A;
	C.ref( all(), tail(-B.n()) ) = B;

	return C;
}

// vertical concat
mat::matrix mat::operator|( const mat::matrix &A, const mat::matrix &B )
{
	if( A.n() != B.n() ) return matrix::ERROR;

	mat::matrix C(A.m() + B.m(), A.n() );

	C.ref( head( A.m()), all() ) = A;
	C.ref( tail(-B.m()), all() ) = B;

	return C;
}

