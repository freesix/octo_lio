#pragma once

#include <lio_sam/nanoflann.hpp>

#include <vector>
/*
    本代码主要实现的是调用nanoflan库处理存储在std::vector中的多维点数据。以便构造kd树。支持下列两种数据
*/
// ===== This example shows how to use nanoflann with these types of containers: =======
//typedef std::vector<std::vector<double> > my_vector_of_vectors_t;
//typedef std::vector<Eigen::VectorXd> my_vector_of_vectors_t;   // This requires #include <Eigen/Dense>
// =====================================================================================


/** A simple vector-of-vectors adaptor for nanoflann, without duplicating the storage.
  *  The i'th vector represents a point in the state space.
  *
  *  \tparam DIM If set to >0, it specifies a compile-time fixed dimensionality for the points in the data set, allowing more compiler optimizations.
  *  \tparam num_t The type of the point coordinates (typically, double or float).
  *  \tparam Distance The distance metric to use: nanoflann::metric_L1, nanoflann::metric_L2, nanoflann::metric_L2_Simple, etc.
  *  \tparam IndexType The type for indices in the KD-tree index (typically, size_t of int)
  */
/**
 * @brief 模板参数
 * @param VectorOfVectorsType 数据类型，如上
 * @param num_t 数据值类型 
 * @param DIM 数据点的维度，默认-1，在编译时根据输入数据来固定
 * @param Distance 距离度量的方法，默认是欧式距离
 * @param IndexType 索引类型
 */
template <class VectorOfVectorsType, typename num_t = double, int DIM = -1, class Distance = nanoflann::metric_L2, typename IndexType = size_t>
struct KDTreeVectorOfVectorsAdaptor
{
	typedef KDTreeVectorOfVectorsAdaptor<VectorOfVectorsType,num_t,DIM,Distance> self_t;
	typedef typename Distance::template traits<num_t,self_t>::distance_t metric_t;
	typedef nanoflann::KDTreeSingleIndexAdaptor< metric_t,self_t,DIM,IndexType>  index_t;

	index_t* index; //! The kd-tree index for the user to call its methods as usual with any other FLANN index.

	/// Constructor: takes a const ref to the vector of vectors object with the data points
	KDTreeVectorOfVectorsAdaptor(const size_t /* dimensionality */, const VectorOfVectorsType &mat, const int leaf_max_size = 10) : m_data(mat)
	{
		assert(mat.size() != 0 && mat[0].size() != 0);
		const size_t dims = mat[0].size();
		if (DIM>0 && static_cast<int>(dims) != DIM)
			throw std::runtime_error("Data set dimensionality does not match the 'DIM' template argument");
		index = new index_t( static_cast<int>(dims), *this /* adaptor */, nanoflann::KDTreeSingleIndexAdaptorParams(leaf_max_size ) );
		index->buildIndex();
	}

	~KDTreeVectorOfVectorsAdaptor() {
		delete index;
	}

	const VectorOfVectorsType &m_data;

	/** Query for the \a num_closest closest points to a given point (entered as query_point[0:dim-1]).
	  *  Note that this is a short-cut method for index->findNeighbors().
	  *  The user can also call index->... methods as desired.
	  * \note nChecks_IGNORED is ignored but kept for compatibility with the original FLANN interface.
	  */
	inline void query(const num_t *query_point, const size_t num_closest, IndexType *out_indices, num_t *out_distances_sq, const int nChecks_IGNORED = 10) const
	{
		nanoflann::KNNResultSet<num_t,IndexType> resultSet(num_closest);
		resultSet.init(out_indices, out_distances_sq);
		index->findNeighbors(resultSet, query_point, nanoflann::SearchParams());
	}

	/** @name Interface expected by KDTreeSingleIndexAdaptor
	  * @{ */

	const self_t & derived() const {
		return *this;
	}
	self_t & derived()       {
		return *this;
	}

	// Must return the number of data points
	inline size_t kdtree_get_point_count() const {
		return m_data.size();
	}

	// Returns the dim'th component of the idx'th point in the class:
	inline num_t kdtree_get_pt(const size_t idx, const size_t dim) const {
		return m_data[idx][dim];
	}

	// Optional bounding-box computation: return false to default to a standard bbox computation loop.
	//   Return true if the BBOX was already computed by the class and returned in "bb" so it can be avoided to redo it again.
	//   Look at bb.size() to find out the expected dimensionality (e.g. 2 or 3 for point clouds)
	template <class BBOX>
	bool kdtree_get_bbox(BBOX & /*bb*/) const {
		return false;
	}

	/** @} */

}; // end of KDTreeVectorOfVectorsAdaptor