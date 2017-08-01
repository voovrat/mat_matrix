#include "mat_matrix.h"


#include <stdio.h>
#include <utility>
#include <iostream>
#include <math.h>

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
			os << s;
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



// +++++++++++++++++++++ arithmetics

// #define POINTWISE_BINOP(fn,)

// void  MAT::operator+=( const mat::matrix & B)
// {
// 	if( m() != B.m() || n() != B.n() ) 
// 		*this = ERROR;
// 	else
// 	{ 
// 		size_t n = B.len();
// 		for( size_t i=0; i < n; i++ ) 
// 			vData[i] += B(i);
//     }
// }

// void  MAT::subBA( const mat::matrix &B); // *this  = B - *this;
// {
// 	if( m() != B.m() || n() != B.n() )
// 		*this = ERROR;
// 	else
// 	{
// 		size_t n=len();
// 		for( size_t i=0; i<n; i++)
// 			vData[i] = B(i) - vData[i];
// 	}

// }


// --- 


void fillMatrix( mat::matrix &A )
{
	size_t m = A.m();
	size_t n = A.n();

	for( size_t i=0; i<m; i++)
	for( size_t j=0; j<n; j++)
		A(i,j) = (i+1)*10  + j+1;
}

void showMatrix( const mat::matrix &A )
{
	std::cout << A;
}


void fillVector( mat::matrix &A)
{

	size_t len = A.len();

	for( size_t i=0; i<len; i++) A(i) = i+3;
}

void showVector( const mat::matrix &A)
{
	size_t len = A.len();
	for( size_t i=0; i<len; i++ )
		printf("%8.3lf",A(i));

	printf("\n");
}


class Mu : public mat::matrix 
{
public:
	Mu(uint  _m = 0,uint  _n = 0, double fill = 0.0)
	: mat::matrix(_m, _n, fill)
	{
	//	setMN(_m,_n);
	}

	Mu(uint  _m, uint  _n, const std::function< double (uint ,uint ) > &fn)
	: mat::matrix(_m,_n,fn)
	{
		//apply([&fn](uint  i,uint  j,double zero){return fn(i,j);} );
	}


	Mu( matrix && B )
	: mat::matrix(std::move(B))
	{
	//	MAT_DBG( this << "( && " << &B << ")" )
	//	swap( *this, B);
	}


    Mu( const mat::matrix &B)
    : mat::matrix(B)
    {
		
    }

    Mu( std::vector<double > && v)
    : mat::matrix(std::move(v))
    {
    
    //	init(std::move(v));
    }



	Mu( const std::vector<double> & v)
    : mat::matrix( std::vector<double>(v) )
    {  }

    Mu( const std::vector<int> &ivec )
    : mat::matrix(ivec)
    {
    	// std::vector<double> dvec( ivec.size() );
    	// for( uint  i=0; i<ivec.size(); i++ )
    	// 	dvec[i] = ivec[i];
    	// init( std::move(dvec) );
    }

    Mu( const std::initializer_list<double>  & list )
    : mat::matrix( list )
    {  }

};

int main()
{
	//mat::matrix A(2,4);

	// std::vector<double> v{1,2,3,4,5};
	// std::cout << apply( mat::matrix {1,2,83,4} , [](double x){ return x+4;} );


	// mat::matrix Z;
	// std::cout << "Z: " << &Z << "\n";
	// Z = mat::construct::zeros(3,3);


	// mat::matrix E = mat::construct::eye(3);
	// std::cout << "E: "<< &E << "\n";

	// mat::matrix _1 = mat::construct::ones(3,3);
	// std::cout << "_1: " << &_1 << "\n";
    

	// std::cout << E << Z << _1;


	// mat::matrix D = ( 2 -  _1 / (_1 + _1) * (E + E + E) * 5  - 1 );
 // 	//mat::matrix D = 5 * E;

	// mat::matrix X = apply(_1,E,[](double x,double y)->double{ return (x + 2) * (y + 3); });

 //    std::cout << "D:" << &D << "\n";
	// std::cout << D;
	// std::cout << X;

	// std::cout << (2*_1).reduce( [](double S, double x)->double{ return S * x; }, 1.0 ) << "\n";

	// std::cout <<  X({0,2},{1,2,0,0}); 

	// std::cout << mat::matrix( iseq(0,10));


	Mu M(10,10, [](uint i, uint j ){ return 100*i+j;} );

	std::cout << M;
	//std::cout << M( iseq(1,10,2), iseq(1,10,2) );

    std::vector<int> I{1,4,5};

    Mu IM{12,15,17};
    std::vector<int> idata = IM.idata();

    std::cout << idata[0] << " öö " << idata[1] << " öö " << idata[2] << "\n";

    std::cout << M(I);
    std::cout << M( IM );
    std::cout << M( {22,11,34} );
    std::cout << M(1) << M(4) << M(5) << "\n";


    std::cout << M( {1,3,5}, {2,4,7} );
    std::cout << M( head(3), tail(-5) );
    std::cout << M( all(), 3);
    std::cout << M( {1,4}, all() );


    M.ref(head(2),head(2)) = M(tail(8),tail(8));
    M.ref({1,2},{1,3}) = 0;
    M.ref( all(), 5) = 1;
    M.ref( 4, all() ) = 3;

    std::cout << M;

    matrix B = -M;

    std::cout << B;

    matrix C = -(M+3);

    std::cout << C;


    matrix X(3,4,[](uint i,uint j){return i*10+j;});

    std::cout << hrep(X,3);
    std::cout << vrep(X,3);

    std::cout << M( tail(-2), tail(-2));




    std::cout << ( ones(3,3) & eye(3) & zeros(3,3) );
    
    mat::matrix Z(3,3,0);
    mat::matrix E = eye(3);
    mat::matrix N(3,3,1);

    std::cout << ( E & Z & N 
                 | Z & N & Z
                 | N & Z & E   
         	     );

    std::vector<double> x{1,2,3};

    mat::matrix v(x);

    v += x;


    std::cout << ( M <= 500);

    matrix U(2,2,{2,4,5,6});


    std::cout << ( U - ( U+1e-15 ) );

    if( bool(U < (U+1e-15) ) ) 
    	std::cout << "U<10!\n";


    matrix I2 = M.find( [](double x){ return x > 500; } );

    std::cout << I2;

    M.ref( [](double x){ return x<500; }) = 11;
    M.refeq(11) = 66;

    matrix W{600};

    std::cout << "UU" << matrix( W.findbetween(600,600));

    M.refbetween(600,800) = 0;

    std::cout << M;

//	std::cout << M( all(), tail(5) ) << M( all(), vec(I) );

 //	A(:, 2:end )
 //	A( seq())
	//fillVector(A);
	//showMatrix(A);

	//fillMatrix(A);
	//showVector(A);

 	return 0;
}