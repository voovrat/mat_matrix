#ifndef MAT_MATRIX_H
#define MAT_MATRIX_H

#include <vector>
#include <unistd.h> // uint 

#include <limits>
#include <ostream>
#include <istream>
#include <utility>
#include <functional>
#include <math.h>

#include <iostream>
#include <stdio.h>

#define MAT_DBG(x)  std::cout<<x<<"\n";
//#define MAT_DBG(x)

namespace mat {


class matrix;

void swap( matrix &A, matrix &B);


namespace range {


// actually this is what i have to use in absence of concepts
// iterable index is just "anything that have uint  operator[](uint  ) and functions uint  size()"
template<class Idx>
class iterable_index // iterable index constrain
{
private:
	const Idx mIndex; 

public:

	iterable_index( const Idx & t) : mIndex(t) {}
	iterable_index( Idx && t) : mIndex(t) {}

	int operator[]( uint  i) const { return mIndex[i]; }
	uint  size() const { return mIndex.size(); }
};



//specialization for vector (i think we can keep the reference instead of copying)
template <>
class iterable_index< std::vector<int> >
{
private:
		  std::vector<int>  mVec;
	const std::vector<int> & rVec;
public:

	iterable_index() : rVec(mVec) {}

	// need to define copy constructors, otherwise our rVec can point to wrong position
	iterable_index( iterable_index< std::vector<int> > && idx ) : mVec(idx.mVec), rVec(mVec) {}
	iterable_index( const iterable_index< std::vector<int> > &idx) : rVec(idx.rVec) {}

  // here we define conversions from types which can be used to index matrices (apart from ranges)
	iterable_index( const std::vector<int> & vec) : rVec(vec) {}
	iterable_index( std::vector<int> && vec) : mVec(vec), rVec(mVec) {}
	iterable_index( const matrix & m );  // is defined right after matrix definition
	iterable_index( const std::initializer_list<int> & list) : mVec(list), rVec(mVec) {}

	int operator[]( uint  i) const { return rVec[i]; }
	uint  size() const { return rVec.size(); } 
};



typedef iterable_index< std::vector<int> > vector_index;

// sequence is just an implementation of iterable sequence with constant step 
struct sequence 
{
	uint  start;
	uint  n;
	uint  step;

	sequence( uint  _start, uint  _n, uint  _step )
	: start(_start), n(_n), step(_step)
	{
		MAT_DBG( "seq " << start << "  " << n << " " << step )
	}

	uint  size() const  {  return n;  }
	int operator[](uint  i) const {  return start + i*step;  }
};


// range is also a constraint, which says that Rng can produce the index of type Idx
template<class IdxStorage, class Generator>
class range 
{
private:
	Generator mGen;
    const Generator &rGen;

public:
	range( const Generator & g) : rGen(g) {}
	range( Generator && g ) : mGen(g), rGen(mGen) {}

	// need to define copy constructors: otherwise when copying rGen can point to a wrong location 
	range( range<IdxStorage,Generator> && r ) 
	: mGen(r.mGen)
	, rGen(mGen)
	{}

	range(const range<IdxStorage,Generator> & r ) : rGen(r.rGen) {}

	const iterable_index<IdxStorage>  index(uint  n) const 
	{
	    return rGen.index(n); 
	}
};


// convert +- notation to real matrix index
// -x means n-x
inline uint  unsigned_index( int i, uint  n)
{
	return i>=0 ? i : n + i;
}


//
//
//  head(n,[step]) - first n 
//  if n<0 --> first size()-n  
//  if step < 0 --> goes backward
class head_gen 
{
	int iMax;
	int iStep;

public:
	head_gen( int mx, int step=1) 
	{
		iMax = mx;
		if( step == 0)
			iStep = 1;
		else
			iStep = step;

	}

	const iterable_index<sequence>  index( uint  n) const 
	{
		uint  _max = unsigned_index(iMax,n);
		uint  step = std::abs(iStep);
		uint  count = std::min( _max + step-1,n+step-1) / step;

		return iterable_index<sequence>( 
                  iStep > 0 ? sequence( 0, count, step) : sequence( _max, count, iStep )
			   );
	}
};

inline const range<sequence,head_gen>  head(int mx,int step=1) 
{
	return range<sequence,head_gen>( head_gen(mx,step) ); 
}


//
//  from first to n
//   if first<0  from n-first to n
//   if step<0 --> from max to min
class tail_gen
{
	int iFirst;
	int iStep;

public:
	tail_gen( int first, int step=1) : iFirst(first), iStep( step==0 ? 1 : step ) {}

	const iterable_index<sequence> index( uint  n) const 
	{   uint  _min = unsigned_index(iFirst,n);
		uint  step = std::abs(iStep);
		uint  count = (n - _min + step-1)/step;


		return iterable_index<sequence>(
                  iStep > 0 ? sequence(_min,count,step) : sequence(n-1,count,-1)
			   );
	}
};

inline const range<sequence,tail_gen> tail(int first,uint  step=1) 
{
	return range<sequence,tail_gen>( tail_gen(first,step) ); 
}


class mid_gen
{
	int iMin;
	int iMax;
	int iStep;

public:
	mid_gen( int mi, int ma, int step=1) : iMin(mi), iMax(ma), iStep(step==0 ? 1 : step ) {}

	const iterable_index<sequence> index( uint  n) const 
	{      
		uint  _mi = unsigned_index(iMin,n);
		uint  _ma = unsigned_index(iMax,n);

       // TODO: read how to do this with std::minmax
		uint  mi = std::min(_mi,_ma);
		uint  ma = std::max(_mi,_ma);

		uint  step = std::abs(iStep);
		uint  count = (ma  - mi + step )/step;

		return iterable_index<sequence>( 
			      iStep>0 ? sequence(mi,count, step ) : sequence(ma,count,iStep) 
			   ); 
	}
};

inline const range<sequence,mid_gen> mid(uint  mi,uint  ma,uint  step=1) 
{
	return range<sequence,mid_gen>( mid_gen(mi,ma,step) ); 
}

class all_gen
{
public:
	all_gen() {}

	const iterable_index<sequence> index(uint  n) const 
	{
		return iterable_index<sequence>( sequence(0,n,1) );
	}
};

inline const range<sequence,all_gen> all() 
{
	return range<sequence, all_gen >( all_gen() );
}



class vector_gen 
{
	vector_index mVecIdx;
	const vector_index &rVecIdx;
public:

	vector_gen( const std::vector<int> & vec) : mVecIdx(vec),rVecIdx(mVecIdx) {}
	vector_gen( const vector_index & vec_idx) : rVecIdx(vec_idx) {}
 	vector_gen( vector_index && vec_idx ) : mVecIdx(vec_idx), rVecIdx(mVecIdx) {}


	const vector_index  & index( uint  n) const 
	{
		return rVecIdx;
	}
};

inline const range<std::vector<int>,vector_gen> vec(const std::vector<int> &v) 
{
	return range<std::vector<int>,vector_gen>( vector_gen(v) ); 
}

inline const range<std::vector<int>,vector_gen> vec(const vector_index &v) 
{
	return range<std::vector<int>,vector_gen>( vector_gen(v) ); 
}


}// namespace range







class matrix 
{
friend void swap(matrix &A, matrix &B);

protected:

	std::vector<double> vData;
    uint  iM,iN,iMN;


	class mref {
	public:
		std::vector<double *> vData; // it important to call the vector the same as in main class (vData) 
									 // cause when we can write generaliyed indexing macros 

		mref( uint n ) : vData(n) {}
		mref( uint m, uint n ) : vData( m*n ) {}
		mref( matrix & M, const std::vector<int> &idx);  // for(i : idx ) vData[i] = &M.data(i)
		
		//	double *& operator[](int i) { return vRef[i]; }

		void operator=( const matrix & m); // { (*this) = m.data(); }
		void operator=( const std::vector<double> & vec ); //  assert(vRef.size() == vec.size() ); for(r : vRef, d:vec) *r = d; 
		void operator=( double d );
	};





public:

	static const matrix ERROR; // error matrix: returned if matrix operations  fail 
	static constexpr double tol = 1.0e-14;  // tolerance 


private:
    double dErrorLvalue; // is used as output when lvalue double & indexing returns error
                         // here points the reference to avoid segmentation fault problems 
    const double dErrorRvalue = std::numeric_limits<double>::quiet_NaN(); 
       // returned NaN. member functions can check : if &result  == &dErrorRvalue then --> there were an error

	void setMN(int _m, int _n)
	{
		iM = _m; 
		iN = _n;
		iMN = _m * _n;
	}

	inline static uint pair_index(uint row, uint col, uint m) 
	{
		return  col * m + row;
	}



public:

	matrix(uint  _m = 0,uint  _n = 0, double fill = 0.0)
	: vData(_m*_n, fill )
	{
		MAT_DBG( this << "(  " << _m << "," << _n << ")" )
		setMN(_m,_n);
	}

	matrix(uint  _m, uint  _n, const std::function< double (uint ,uint ) > &fn)
	: matrix(_m,_n,0.0)
	{
		MAT_DBG( this << "(  " << _m << "," << _n << ", fn )" )
		apply([&fn](uint  i,uint  j,double zero){return fn(i,j);} );
	}

	matrix(uint _m, uint _n, const std::vector<double> &v)
	{
		if( v.size() != _m * _n ) 
			*this = ERROR; 
		else {
			vData = v;
			setMN(_m,_n);
		}

	}


	matrix( matrix && B )
	{
		MAT_DBG( this << "( && " << &B << ")" )
		swap( *this, B);
	}


    matrix( const mat::matrix &B)
    {
		MAT_DBG( this << "( & " << &B << ")" )

 		vData = B.vData;
 		iM = B.iM;
 		iN = B.iN;
 		iMN = B.iMN;
    }

    matrix( std::vector<double > && v)
    {
    	init(std::move(v));
    }
	
    matrix( const std::vector<int> & v)
    {
       	vData.assign(v.begin(),v.end());
       	setMN(v.size(),1);
    }


protected:
    void init( std::vector<double> &&v)
    {
    	iM = v.size();
    	iN = 1;
    	iMN = iM;
    	swap( vData, v);
    }

public:


    matrix( const std::vector<double> & v)
    : matrix( std::vector<double>(v) )
    {  }


    matrix( const std::initializer_list<double>  & list )
    : matrix( std::vector<double>(list) )
    {  }



    virtual ~matrix() 
    {
    	MAT_DBG( "~" << this << " : " << len() << " . " << vData.size() )
    }

    // better return nothing --> we are sure that no "ghost" objects are created and immedeately deleted
	void operator=( const mat::matrix & B)
	{
		MAT_DBG( this << "= &" << &B )

		mat::matrix C(B);
		swap(*this,C);
	}

	void operator=( mat::matrix && B )
	{
		MAT_DBG( this << "= &&" << &B )
		swap(*this,B);
	}


	// conversion to bool:  matrix is True, when all elements are 1
	// in this way we can write for example if(A == B ) {... }
	explicit operator bool()
	{
		for( double x : vData ) 
			if( fabs(x-1)>tol) return false;
		return true;
	}



	// ---  size  operations 

    uint  m() const { return iM;}
    uint  n() const { return iN; }

	void size(uint   &_m, uint  &_n ) const 
	{
		_m = m();
		_n = n(); 

	}

	void resize(uint  _m, uint  _n, double fill = 0.0 )
	{
        setMN(_m,_n);
        vData.assign(iMN,fill);
	}


	uint  len()  const { return iMN; }

	// direct data access

	std::vector<double> &vdata() {return vData; }
	double & vdata( uint i) { return vData[i]; } 

	const std::vector<double> &data() const { return vData;}
 	double data( uint i) const { return vData[i]; }

	double * vpdata() { return &vData[0]; }
	const double *pdata() const { return &vData[0]; }


	std::vector<int> idata() const 
	{
		std::vector<int> ret( len() );
		for( uint  i=0; i<len(); i++ )
			ret[i] = (int)( vData[i] + 0.5 );
		return ret;
	}


    // ****** indexing ********


	// -- simple indexing (i,j)

	double & operator() ( int row, int col) 
	{
		double &ret = (double &)((const matrix*)this)->operator()(row,col);
		if( &ret == &dErrorRvalue )
		{
			*this = ERROR;
			ret = dErrorLvalue;
		}
		return ret;
	}

	const double & operator() (int row, int col) const 
	{
		uint  urow = range::unsigned_index(row,iM);
		uint  ucol = range::unsigned_index(col,iN);

		if ( urow >= iM || ucol >= iN ) 
		     return  dErrorRvalue;
		
		return vData[ ucol * iM + urow ];
	}

	const double & operator() ( int idx ) const
	{
		static constexpr double NaN = std::numeric_limits<double>::quiet_NaN();

		uint  uidx = range::unsigned_index(idx,iMN);

		if (uidx < iMN)
			return vData[uidx];

		return dErrorRvalue;
	}

	double & operator() (int idx) 
	{
		double &r = (double &)((const matrix *)this)->operator()(idx);
		if( &r == &dErrorRvalue ) 
		{
			*this = ERROR;
			r = dErrorLvalue;
		}
		
		return r;
	}


	// ---- vector indexing A(vec)

	#define OPERATOR_IDX( fnName, ClassName, Const, Ref )           \
	template<class Idx>                                             \
	ClassName fnName( const range::iterable_index<Idx> &idx) Const  \
	{                                                               \
		uint  n = idx.size();                                       \
		ClassName A( n, 1 );                                        \
		for( uint  i = 0; i<n; i++ )                                \
			A.vData[i] = Ref (*this)( idx[i] );                     \
		return A;		                                            \
	}

	OPERATOR_IDX( operator(), mat::matrix, const, )
	OPERATOR_IDX( ref,    mat::matrix::mref, , & )


	 #undef OPERATOR_IDX

	// template< class Idx > 
	// mat::matrix operator()(const range::iterable_index<Idx> &idx) const
	// {
	// 	uint  n = idx.size();
	// 	mat::matrix A( n, 1 );
	// 	for( uint  i = 0; i<n; i++ )
	// 		A.vData[i] = (*this)( idx[i] );
	// 	return A;		
	// }


//===


	template< class Idx, class Gen >
	mat::matrix operator()( const range::range<Idx,Gen> &gen) const 
	{
		return (*this)( gen.index(iMN) );
	}

	template<class Idx, class Gen>
	mat::matrix::mref ref( const range::range<Idx,Gen> &gen )
	{
		return this->ref( gen.index(iMN) );
	}


	// we don't need actually the function of vector, initialiyer list etc.
	// we need only one explicit function for iterable_list<vector> - everything else is convertable to it
	mat::matrix operator()( const range::vector_index & idx ) const 
	{
		return this->operator()< std::vector<int> >(idx);
	}

	mat::matrix::mref ref( const range::vector_index & idx ) 
	{
		return this->ref< std::vector<int> >(idx);
	}


	// double vector indexing  A(vec1,vec2)


	#define OPERATOR_IDX_IDX( fnName, ClassName, Const, Ref )             \
	template<class Idx1, class Idx2>                                      \
	ClassName fnName( const range::iterable_index<Idx1> &I                \
		                , const range::iterable_index<Idx2> &J ) Const    \
	{   uint  m = I.size();                                               \
		uint  n = J.size();                                               \
		ClassName A(m,n);                                                 \
        for( uint  i=0; i<m; i++ )                                        \
		for( uint  j=0; j<n; j++ )                                        \
				A.vData[ pair_index(i,j,m) ] = Ref (*this)(I[i], J[j]); \
		return A;                                                         \
	}


	OPERATOR_IDX_IDX( operator(), mat::matrix       , const,   )
	OPERATOR_IDX_IDX( ref       , mat::matrix::mref ,      , & )

    #undef OPERATOR_IDX_IDX


	// template<class Idx1, class Idx2>
	// mat::matrix operator()( const range::iterable_index<Idx1> &I, const range::iterable_index<Idx2> &J ) const 
	// {
	// 	uint  m = I.size();
	// 	uint  n = J.size();
	// 	mat::matrix A(m,n);

	// 	for( uint  i=0; i<m; i++ )
	// 	for( uint  j=0; j<n; j++ )
	// 			A(i,j) = (*this)(I[i], J[j]);

	// 	return A;
	// }

	template<class Idx1, class Gen1, class Idx2, class Gen2 >
	mat::matrix operator()( const range::range<Idx1,Gen1> &I, const range::range<Idx2,Gen2> &J) const 
	{
		return (*this)( I.index(iM), J.index(iN) );
	}

	template<class Idx1, class Gen1, class Idx2, class Gen2 >
	mat::matrix::mref ref( const range::range<Idx1,Gen1> &I, const range::range<Idx2,Gen2> &J) 
	{
		return this->ref( I.index(iM), J.index(iN) );
	}


    // again, don't need to define all posible combinations of indexers
    // need to define>
    //   a) one explicit  operator( iterrable_index<vector>, iterable_index<vector> ) 
    //      for all indexers like ( vector_like , vector_like )
    //
    //   b) two templates  template <Range> operator(  Range<>,  iterable_index<vector> )
    //               and   template <Range> operator( iterable_index<vector>, Range<> )
    //  for all indexers like  (range, vector_like ) and (vector_like, range)

	mat::matrix operator()( const range::vector_index &I, const range::vector_index &J ) const 
	{  
		return this->operator()<std::vector<int>, std::vector<int> >( I, J );
	}

	mat::matrix::mref ref( const range::vector_index &I, const range::vector_index &J ) 
	{  
		return this->ref<std::vector<int>, std::vector<int> >( I, J );
	}



	template< class Idx, class Gen >
	mat::matrix operator()( const range::range<Idx, Gen> &I, const range::vector_index &J ) const 
	{
		return this->operator()< Idx, Gen, std::vector<int>, range::vector_gen >( I, range::vec(J) );
	}


	template< class Idx, class Gen >
	mat::matrix::mref ref( const range::range<Idx, Gen> &I, const range::vector_index &J ) 
	{
		return this->ref< Idx, Gen, std::vector<int>, range::vector_gen >( I, range::vec(J) );
	}



	template< class Idx, class Gen >
	mat::matrix operator()( const range::vector_index &I, const range::range<Idx,Gen> &J ) const 
	{
		return this->operator()< std::vector<int>,  range::vector_gen, Idx, Gen >( range::vec(I), J );
	}

	template< class Idx, class Gen >
	mat::matrix ref( const range::vector_index &I, const range::range<Idx,Gen> &J )  
	{
		return this->ref< std::vector<int>,  range::vector_gen, Idx, Gen >( range::vec(I), J );
	}


	//  row  A(i,vec)


	#define OPERATOR_ROW_IDX( fnName, ClassName, Const, Ref )                 \
	template<class Idx>                                                       \
	ClassName fnName( int i, const range::iterable_index<Idx> &J ) Const  \
	{   uint  n = J.size();                                                   \
		ClassName A(1,n);                                                     \
		for( uint  j=0; j<n; j++)                                             \
			A.vData[j] = Ref (*this)(i, J[j] );                               \
		return A;                                                             \
	}

	OPERATOR_ROW_IDX( operator(), mat::matrix       , const,   )
	OPERATOR_ROW_IDX( ref       , mat::matrix::mref ,      , & )

	#undef OPERATOR_ROW_IDX

	// template<class Idx>
	// mat::matrix operator()( int i, const range::iterable_index<Idx> &J ) const 
	// {
	// 	uint  n = J.size();
	// 	mat::matrix A(1,n);

	// 	for( uint  j=0; j<n; j++)
	// 		A(j) = (*this)(i, J[j] );
	// 	return A;
	// }


	template< class Idx, class Gen >
	mat::matrix operator()( int  i, const range::range<Idx,Gen> &J) const 
	{
		return (*this)(i,J.index(iN));
	}

	template< class Idx, class Gen >
	mat::matrix::mref ref( int  i, const range::range<Idx,Gen> &J) 
	{
		return this->ref(i,J.index(iN));
	}


	mat::matrix operator()( int  i, const range::vector_index &J) const 
	{
		return this->operator()< std::vector<int> >(i,J);
	}
	

	mat::matrix::mref ref( int  i, const range::vector_index &J) 
	{
		return this->ref< std::vector<int> >(i,J);
	}

	//  col  A(vec,j)

	#define OPERATOR_IDX_COL( fnName, ClassName, Const, Ref )              \
	template<class Idx>                                                    \
	ClassName fnName( const range::iterable_index<Idx> &I, int  j ) Const  \
	{	uint  m = I.size();                                                \
		ClassName A(m,1);                                                \
		for( uint  i=0; i<m; i++ )                                         \
				A.vData[i] = Ref (*this)(I[i], j );                        \
		return A;                                                          \
	}

	OPERATOR_IDX_COL( operator(), mat::matrix       , const ,   )
    OPERATOR_IDX_COL( ref       , mat::matrix::mref ,       , & )

    #undef OPERATOR_IDX_COL


	// template<class Idx>
	// mat::matrix operator()( const range::iterable_index<Idx> &I, int  j ) const 
	// {
	// 	uint  m = I.size();
	// 	mat::matrix A(m,1);

	// 	for( uint  i=0; i<m; i++ )
	// 			A(i) = (*this)(I[i], j );
	// 	return A;
	// }

	template<class Idx, class Gen >
	mat::matrix operator()( const range::range<Idx,Gen> &I, int  j) const 
	{
		return (*this)(I.index(iM),j);
	}

	template<class Idx, class Gen >
	mat::matrix::mref ref( const range::range<Idx,Gen> &I, int  j) 
	{
		return this->ref(I.index(iM),j);
	}


	mat::matrix operator()( const range::vector_index &I, int j) const 
	{
		return this->operator()< std::vector<int> >(I,j);
	}

	mat::matrix::mref ref( const range::vector_index &I, int j) 
	{
		return this->ref< std::vector<int> >(I,j);
	}


	// find

	std::vector<int>  find( const std::function<bool (double)> &fn ) const 
	{
		std::vector<int> v;
		v.reserve(iMN);
		for( uint i = 0; i< iMN; i++ )
			if( fn(vData[i]) ) v.push_back(i);
		return v;
	}

	mref ref( const std::function<bool (double)> &fn )
	{
		return mref(*this,find(fn));
	}

	
	#define MAT_FIND_CMP( opname, comp )                                                            \
	std::vector<int> find##opname( double d) const { return find( [d](double x){ return comp; }); } \
	mref ref##opname( double d) { return mref(*this,find##opname(d)); }                             \


	MAT_FIND_CMP( eq,     (fabs(x-d)<tol) )
	MAT_FIND_CMP( neq,    (fabs(x-d)>tol) )
	MAT_FIND_CMP( less,   ( x < d-tol ) )
    MAT_FIND_CMP( lesseq, ( x < d+tol ) )
    MAT_FIND_CMP( more,   ( x >  d+tol ) )
    MAT_FIND_CMP( moreeq, ( x > d-tol) )

    #undef MAT_FIND_CMP

    std::vector<int> findbetween(double mi, double ma)
    {
    	return find([mi,ma](double x) { return (x>=mi-tol) && (x<=ma+tol); } );
    }

    mref refbetween( double mi, double ma)
    { return mref(*this,findbetween(mi,ma));}

    // ---- in/out

	void print( std::ostream & os, const char * fmt = "%8.3lg") const;



	// ++++ arithmetics : 


    // unary

	// transpose
    matrix T( ) const 
    {
    	uint _m = m();
    	uint _n = n();

    	matrix B( _n, _m );
    	for( uint i=0; i<_m; i++ )
    	for( uint j=0; j<_n; j++ )
    		B(j,i) = (*this)(i,j);

    	return B;
    }

    void neg() 
    {
    	for(uint i=0; i<iMN; i++ ) vData[i] = -vData[i];
    }


	matrix operator-() const &
	{
		MAT_DBG( this << "operator- const &" )
		matrix A(*this);
		A.neg();
		return A;
	}

	matrix  operator-() &&
	{
		MAT_DBG( this << "operator- && "  )
		neg();
		return *this;
	}


    // pointwise operation

	template<class Op>
	void pointwise( const mat::matrix &B)
	{
		if( m() != B.m() || n() != B.n() ) 
			*this = ERROR;
		else
		{ 
			uint  n = B.len();
			for( uint  i=0; i < n; i++ ) 
				Op::fn(vData[i], B(i) );
	    }
	}


	// constant operation
	template<class Op>
	void constop( double c)
	{
		uint  n = vData.size();
		for( uint  i=0; i < n; i++ ) 
				Op::fn(vData[i], c );
	}	

	// operations 
	struct OpPlus {
		 static inline void fn( double &a, const double &b) { a+=b; }
	};

	struct OpMinus {
		 static inline void fn( double &a, const double &b) { a-=b; }
	};

	struct OpMinusBA {
		 static inline void fn( double &a, const double &b) { a=b-a; }
	};

	struct OpMul {
			 static inline void fn( double &a, const double &b) { a*=b; }
	};
	struct OpDiv {
		 static inline void fn( double &a, const double &b) { a/=b; }
	};
	struct OpDivBA {
		 static inline void fn( double &a, const double &b) { a=b/a; }
	};




	// conditional ops
	struct OpEq {
		static inline void fn( double &a, const double &b ) { a = (double)(fabs(a-b)<tol); }
	};

	struct OpNeq {
		static inline void fn( double &a, const double &b ) { a = (double)(fabs(a-b)>=tol ); }
	};

	struct OpLess {
		static inline void fn( double &a, const double &b ) { 
			MAT_DBG( a-b << " <? " << -tol );
			a = (double)(  a - b < -tol ); }
	};

	struct OpLessEq {
		static inline void fn( double &a, const double &b ) { a = (double)(  a - b <=  tol ); }
	};


	struct OpMore {
		static inline void fn( double &a, const double &b ) { a = (double)(a - b > tol ); }
	};

	struct OpMoreEq {
		static inline void fn( double &a, const double &b ) { 
			MAT_DBG( a-b << " >=? " << -tol );
			a = (double)(a - b >= - tol ); }
	};

	// all pointwise and constant += operations

    // .....  += 

	void  operator+=( const mat::matrix & B) 
	{
		MAT_DBG( this << "+=" << &B )
		pointwise<OpPlus>(B);
		//pointwise< [](double &a, double &b){a+=b} >(B);
	}

	void operator+=( double c)
	{
		MAT_DBG( this << "+=" << c )
		constop<OpPlus>(c);	
	}

    // ....... -= 

	void  operator-=(const mat::matrix &B)
	{
		MAT_DBG( this << "-=" << &B )
		pointwise<OpMinus>(B);		
	}

	void  operator-=( double c)
	{
		MAT_DBG( this << "-=" << c )
		constop<OpMinus>(c);		
	}

	// ....   *this  = B - *this;
	void  subBA( const mat::matrix &B) 
	{
		MAT_DBG( this << "=" << &B << "-" << this )
		pointwise<OpMinusBA>(B);	
	}
	
	void  subBA( double c ) 
	{
		MAT_DBG( this << "=" << c << "-" << this )
		constop<OpMinusBA>(c);	
	}

	// ....  pointwise multiplication
	void operator*=(const mat::matrix &B)
	{
		MAT_DBG( this << "*=" << &B )
		pointwise<OpMul>(B);		
	}

	void operator*=( double c)
	{
		MAT_DBG( this << "*=" << c )
		constop<OpMul>(c);		
	}

	// ..... pointwise division
	void operator/=(const mat::matrix &B)
	{
		MAT_DBG( this << "/=" << &B )
		pointwise<OpDiv>(B);		
	}

	void operator/=( double c )
	{
		MAT_DBG( this << "/=" << c )
		constop<OpDiv>(c);		
	}

	// .... pointwise backwards division: this = B / this
	void divBA(const mat::matrix &B)
	{
		MAT_DBG( this << "=" << &B << "/" << this )
		pointwise<OpDivBA>(B);		
	}

	void divBA( double  c )
	{
		MAT_DBG( this << "=" << c << "/" << this )
		constop<OpDivBA>(c);		
	}

	// the next operations are "inplace", so the result of comparison replace the data in the matrix itself 


	// ==
	void inplaceEq( const mat::matrix &B)
	{
		MAT_DBG( this << "= " << "(" << this << " == " << &B << ")" );
		pointwise<OpEq>(B);
	}

	void inplaceEq( double c)
	{
		MAT_DBG( this << "= " << "(" << this << " == " << c << ")" );
		constop<OpEq>(c);
	}


	// !=
	void inplaceNeq( const mat::matrix &B)
	{
		MAT_DBG( this << "= " << "(" << this << " != " << &B << ")" );
		pointwise<OpNeq>(B);
	}

	void inplaceNeq( double c)
	{
		MAT_DBG( this << "= " << "(" << this << " != " << c << ")" );
		constop<OpNeq>(c);
	}


	// <
	void inplaceLess( const mat::matrix &B)
	{
		MAT_DBG( this << "= " << "(" << this << " < " << &B << ")" );
		pointwise<OpLess>(B);
	}

	void inplaceLess( double c)
	{
		MAT_DBG( this << "= " << "(" << this << " < " << c << ")" );
		constop<OpLess>(c);
	}


	// <=
	void inplaceLessEq( const mat::matrix &B)
	{
		MAT_DBG( this << "= " << "(" << this << " <= " << &B << ")" );
		pointwise<OpLessEq>(B);
	}

	void inplaceLessEq( double c)
	{
		MAT_DBG( this << "= " << "(" << this << " <= " << c << ")" );
		constop<OpLessEq>(c);
	}


	// >
	void inplaceMore( const mat::matrix &B)
	{
		MAT_DBG( this << "= " << "(" << this << " > " << &B << ")" );
		pointwise<OpMore>(B);
	}

	void inplaceMore( double c)
	{
		MAT_DBG( this << "= " << "(" << this << " > " << c << ")" );
		constop<OpMore>(c);
	}


	// >=
	void inplaceMoreEq( const mat::matrix &B)
	{
		MAT_DBG( this << "= " << "(" << this << " >= " << &B << ")" );
		pointwise<OpMoreEq>(B);
	}

	void inplaceMoreEq( double c)
	{
		MAT_DBG( this << "= " << "(" << this << " >= " << c << ")" );
		constop<OpMoreEq>(c);
	}


	// apply functions 

	void apply( const std::function< double (double)> & fn )
	{
		for( double &x : vData ) 
			x = fn(x);
	}

	void apply( const std::function< double (uint , uint , double ) > &fn )
	{
		for(uint  i=0; i<m(); i++)
		for(uint  j=0; j<n(); j++)
		{
			(*this)(i,j) = fn( i,j, (*this)(i,j) );
		}
	}

	void apply( const mat::matrix &B, const std::function< double (double,double) > &fn )
	{
		if( m() != B.m() || n() != B.n() )
			*this = ERROR;
		else
		{
			uint  n = len();
			for( uint  i =0; i<n; i++)
				vData[i] = fn( vData[i],B(i));
		}
	}

	void apply( const matrix &B, const std::function< double (uint , uint , double, double ) > &fn )
	{
		for(uint  i=0; i<m(); i++)
		for(uint  j=0; j<n(); j++)
		{
			(*this)(i,j) = fn( i,j, (*this)(i,j), B(i,j));
		}
	}

	void apply( double d, const std::function< double (double,double) > &fn )
	{
		for( double &x : vData )
			x = fn(x,d);
	}

	void apply( double d, const std::function< double (uint , uint , double, double ) > &fn )
	{
		for(uint  i=0; i<m(); i++)
		for(uint  j=0; j<n(); j++)
		{
			(*this)(i,j) = fn( i,j, (*this)(i,j), d);
		}
	}


	double reduce( const std::function<double (double,double)> &fn, double S0 = 0.  ) const 
	{
		double S = S0;
		for( const double &x : vData )
			S = fn(S,x);
		return S;
	}

};


inline void mat::matrix::mref::operator=( const matrix & m)
{
	*this = m.data();
}

inline void mat::matrix::mref::operator=( const std::vector<double> & vec )
{
	uint n = std::min( vData.size(),  vec.size() );
  	for(uint i=0; i<n; i++)
  		*vData[i] =  vec[i];
}

inline void mat::matrix::mref::operator=( double d )
{
	for( double * px : vData) *px = d;
}

inline mat::matrix::mref::mref( matrix & M, const std::vector<int> &idx)  // vData[i] = &M.data(i)
: vData( idx.size() )
{
	uint n = idx.size();
	for( uint i=0; i<n; i++)
		vData[i] = &M.vdata( idx[i] );
}



namespace range {

//template <>
//iterable_index< std::vector<uint > >
iterable_index< std::vector<int > >::iterable_index( const matrix &m ) 
  : mVec( std::move(m.idata() ) )
  , rVec(mVec) 
{
	MAT_DBG( "mat2vec : " << rVec.size() << " " << rVec[0] )

}

}




void swap( matrix &A, matrix &B)
{
	std::swap( A.vData, B.vData);
	std::swap( A.iM, B.iM);
	std::swap( A.iN, B.iN);
	std::swap( A.iMN, B.iMN);
}



std::ostream & operator<<( std::ostream & os, const matrix & A)
{
	A.print(os);
	return os;
}

// pointwise operations 

// we have 3 versions of each operation:  & op &, && op &, & op &&, for faster calcs
// we define differently the commutative operations and non-commutative ops

#define MAT_DEFINE_POINTWISE_OP(op,fwdop,backop)              \
mat::matrix op(const mat::matrix &_A, const mat::matrix &B) { \
   MAT_DBG( "& " << &_A << " " #op " & " << &B )              \
	mat::matrix A(_A);                                        \
	fwdop;                                                    \
	return A;                                                 \
}                                                             \
                                                              \
mat::matrix && op( const mat::matrix &A, mat::matrix &&B) {   \
    MAT_DBG( "&" << &A << " " #op " && " << &B )              \
	backop;                                                   \
    return std::move(B);                                      \
}                                                             \
                                                              \
mat::matrix && op( mat::matrix &&A, const mat::matrix &B ) {  \
    MAT_DBG( "&&" << &A << " " #op " & " << &B )              \
    fwdop;                                                    \
	return std::move(A);                                      \
}                                                             \
mat::matrix && op( mat::matrix &&A, mat::matrix &&B ) {       \
    MAT_DBG( "&&" << &A << " " #op " && " << &B )             \
	fwdop;                                                    \
	return std::move(A);                                      \
}    


MAT_DEFINE_POINTWISE_OP(operator+,A+=B, B+=A )
MAT_DEFINE_POINTWISE_OP(operator-,A-=B, B.subBA(A) )
MAT_DEFINE_POINTWISE_OP(operator*,A*=B, B*=A )
MAT_DEFINE_POINTWISE_OP(operator/,A/=B, B.divBA(A) )

MAT_DEFINE_POINTWISE_OP(operator==, A.inplaceEq(B)    , B.inplaceEq(A)  )
MAT_DEFINE_POINTWISE_OP(operator!=, A.inplaceNeq(B)   , B.inplaceNeq(A) )
MAT_DEFINE_POINTWISE_OP(operator< , A.inplaceLess(B)  , B.inplaceMore(A) )
MAT_DEFINE_POINTWISE_OP(operator<=, A.inplaceLessEq(B), B.inplaceMoreEq(A) )
MAT_DEFINE_POINTWISE_OP(operator> , A.inplaceMore(B)  , B.inplaceLess(A) )
MAT_DEFINE_POINTWISE_OP(operator>=, A.inplaceMoreEq(B), B.inplaceLessEq(A) )


#undef MAT_DEFINE_POINTWISE_OP

#define MAT_DEFINE_CONST_OP(op,fwdop,backop)        \
mat::matrix op(const mat::matrix &_A, double  d ) { \
   MAT_DBG( "& " << &_A << " " #op "  " << d )      \
	mat::matrix A(_A);                              \
	fwdop;                                          \
	return A;                                       \
}                                                   \
mat::matrix op(double  d, const mat::matrix &_A ) { \
   MAT_DBG(  d << " " #op " & " << &_A )            \
	mat::matrix A(_A);                              \
	backop;                                         \
	return A;                                       \
}                                                   \
mat::matrix && op( mat::matrix &&A, double  d ) {   \
   MAT_DBG( "&& " << &A << " " #op "  " << d )      \
	fwdop;                                          \
	return std::move(A);                            \
}                                                   \
mat::matrix && op(double  d,  mat::matrix &&A ) {   \
   MAT_DBG( d << " " #op " && " << &A )             \
	backop;                                         \
	return std::move(A);                            \
}                                                   \

MAT_DEFINE_CONST_OP(operator+,A+=d,A+=d)
MAT_DEFINE_CONST_OP(operator-,A-=d,A.subBA(d))
MAT_DEFINE_CONST_OP(operator*,A*=d,A*=d)
MAT_DEFINE_CONST_OP(operator/,A/=d,A.divBA(d))

MAT_DEFINE_CONST_OP(operator==, A.inplaceEq(d)    , A.inplaceEq(d)  )
MAT_DEFINE_CONST_OP(operator!=, A.inplaceNeq(d)   , A.inplaceNeq(d) )
MAT_DEFINE_CONST_OP(operator< , A.inplaceLess(d)  , A.inplaceMore(d) )
MAT_DEFINE_CONST_OP(operator<=, A.inplaceLessEq(d), A.inplaceMoreEq(d) )
MAT_DEFINE_CONST_OP(operator> , A.inplaceMore(d)  , A.inplaceLess(d) )
MAT_DEFINE_CONST_OP(operator>=, A.inplaceMoreEq(d), A.inplaceLessEq(d) )


#undef MAT_DEFINE_CONST_OP


/// concatanation (v and h)

// horizontal concat
mat::matrix operator&( const mat::matrix &A, const mat::matrix &B );

// vertical concat
mat::matrix operator|( const mat::matrix &A, const mat::matrix &B );




//
//  * * * * ** * * ** * * * MATRIX MULTIPLICATION * * * * * * * * * * 
//
//

mat::matrix operator%( const mat::matrix &A, const mat::matrix &B);


//
//  apply 
//

mat::matrix apply(  const mat::matrix &A, const std::function<double (double)> &fn)
{
	mat::matrix B( A );
	B.apply( fn );
	return B;
}

mat::matrix && apply( mat::matrix && A, const std::function<double (double)> &fn) 
{
	A.apply( fn );
	return std::move(A);
}

mat::matrix apply( const mat::matrix &A, const std::function<double (uint , uint , double)> &fn)
{
	mat::matrix B(A);
	B.apply(fn);
	return B;
}

mat::matrix apply(  mat::matrix &&A, const std::function<double (uint , uint , double)> &fn)
{
	A.apply(fn);
	return std::move(A);
}

mat::matrix apply( const mat::matrix &A, const mat::matrix &B, const std::function<double (double,double)> &fn)
{
	mat::matrix C(A);
	C.apply(B,fn);
	return C;
}

mat::matrix apply( const mat::matrix &A, const mat::matrix &B, const std::function<double (uint , uint ,double,double)> &fn)
{
	mat::matrix C(A);
	C.apply(B,fn);
	return C;
}

mat::matrix && apply( mat::matrix &&A, const mat::matrix &B, const std::function<double (double,double)> &fn)
{
	A.apply(B,fn);
	return std::move(A);
}

mat::matrix apply( mat::matrix &&A, const mat::matrix &B, const std::function<double (uint , uint ,double,double)> &fn)
{
	A.apply(B,fn);
	return std::move(A);
}

mat::matrix apply( const mat::matrix &A, double d, const std::function<double (double,double) > &fn)
{
	mat::matrix B(A);
	B.apply( d, fn);
	return B;
}

mat::matrix apply( const mat::matrix &A, double d, const std::function<double (uint , uint , double,double) > &fn)
{
	mat::matrix B(A);
	B.apply( d, fn);
	return B;
}


namespace construct {

	// factories  (constants)
	matrix zeros(int m, int n);
	matrix ones(int m, int n); 
	matrix eye( int n);

	matrix hrep( const matrix &A, int n);
	matrix vrep( const matrix &A, int n);


	template<class T, class TStep = T >
	std::vector<T> seq( T from, T to, TStep step=1.)
	{
		// example:
		// (1,10,2)
		//  (10-1)/2 = 4.5, n = 4!
		//  1 3 5 7  not 9! 

		//  1 1+2 1+2*2  1+3*2  1+4*2 

		//  f  f+s   f+2s  .... f+ns t     - f
		//  0  s     2s         ns  t-f    / s
		//  0  1    2           n   (t-f)/s   

		uint  n = (uint )fabs( (double)(to - from ) / step );


		// std::cout << " to: "      << to 
		//           << " from: "    << from 
		//           << " step: "    << step 
		//           << " (t-f)/s: " <<  ( (double)(to - from + 1) / step )
		//           << " n: "		  << n
		//           << "\n";

		std::vector<T> vec( n + 1 );
		for( T x = from, i=0; i<=n; i++, x+=step )
				vec[i] = x;
		return vec;
	}

	std::vector<int > iseq( uint  from, uint  to, int step = 1)
	{
		return seq<int ,int>(from,to,step);
	}

	std::vector<double> dseq( double from, double to, double step=1. )
	{
		return seq<double>(from,to,step);
	}

	matrix mseq(double from, double to, double step = 1.)
	{
		return matrix( dseq(from, to, step ) );
	}
}


namespace range 
{


}


}

#endif