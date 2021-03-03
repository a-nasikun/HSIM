// Short-Term Vector Dijkstra
// demonstrational implementation
// 01 July 2013

// In case you make use of this method in research work,
// please cite the accompanying publication:

// @article{CHK13,
//  journal = {Computer Graphics Forum, Proc. SGP 2013},
//  title = {{Practical Anisotropic Geodesy}},
//  author = {Marcel Campen and Martin Heistermann and Leif Kobbelt},
//  pages = {63-71},
//  volume= {32},
//  number= {5},
//  year = {2013}}
#pragma once
#ifndef STVD_H
#define STVD_H

#include <OpenMesh/Core/IO/MeshIO.hh>
#include <OpenMesh/Core/Mesh/TriMesh_ArrayKernelT.hh>
#include <OpenMesh/Core/Geometry/VectorT.hh>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <iterator>
#include <limits>
#include <set>
#include <utility>
#include <cmath>

typedef OpenMesh::TriMesh_ArrayKernelT<>  TriMesh;

using OpenMesh::HalfedgeHandle;
using OpenMesh::VertexHandle;
using OpenMesh::Vec2d;
using OpenMesh::Vec3d;

struct EdgeMetric
{
    double angle;
    Vec2d weight;

    friend std::istream& operator>>(std::istream &is, EdgeMetric &em)
    {
        is >> em.angle
           >> em.weight[0]
           >> em.weight[1];
        return is;
    }
};

class STVD
{
public:
    STVD(const TriMesh &_mesh,
            const std::vector<EdgeMetric> &_metric)
        : mesh(_mesh),
          metric(_metric),
          h_accumangle_(mesh.n_halfedges())
    {
        prepare();
    }

    const std::vector<double> &run(int k, int source)
    {
        e.resize(k);
        angles.resize(k);
        weights.resize(k);

        std::set<std::pair<double, int> > q; // priority queue: distance, vertex
        pred.assign(mesh.n_vertices(), HalfedgeHandle(-1));
        is_final.assign(mesh.n_vertices(), false);
        dist.assign(mesh.n_vertices(), std::numeric_limits<double>::infinity());

        dist[source] = 0.0;
        q.insert(std::make_pair(dist[source], source));

        while(!q.empty()) {
            int v = q.begin()->second;
            q.erase(q.begin());
            is_final[v] = true;

            TriMesh::ConstVertexOHalfedgeIter voh_it;
            for (voh_it = mesh.cvoh_iter(VertexHandle(v)); voh_it; ++voh_it) {
                int w = mesh.to_vertex_handle(voh_it).idx(); // adjacent vertex
                if (is_final[w])
                    continue;
                double new_dist = update_dist(v, voh_it.handle(), w, k);
                if (new_dist < dist[w]) {
                    pred[w] = mesh.opposite_halfedge_handle(voh_it);
                    if(dist[w] != std::numeric_limits<double>::infinity())
                        q.erase(std::make_pair(dist[w], w));
                    dist[w] = new_dist;
                    q.insert(std::make_pair(dist[w], w));
                }
            }
        }
        return dist;
    }

private:
    Vec2d polar_to_xy(double len, double angle)
    {
        return Vec2d(-cos(angle) * len, sin(angle) * len);
    }

    double length(Vec2d vec, double angle, Vec2d weight)
    {   // evaluate Riemannian metric for the curve defined by vec
        Vec2d major_axis(-cos(angle), sin(angle));
        Vec2d minor_axis(major_axis[1], -major_axis[0]);
        return Vec2d((vec | major_axis) * weight[0], (vec | minor_axis) * weight[1]).norm();
    }

    void prepare()
    {
        // precompute normalized cumulative angles around each vertex
        TriMesh::VertexIter vend = mesh.vertices_end();
        for(TriMesh::VertexIter vi = mesh.vertices_begin(); vi != vend; ++vi) {
            TriMesh::VertexHandle v = vi.handle();
            double angle_accum = 0.0;
            TriMesh::ConstVertexIHalfedgeIter vih_it;
            for (vih_it = mesh.cvih_iter(VertexHandle(v)); vih_it; ++vih_it) {
                h_accumangle_[vih_it.handle().idx()] = angle_accum;
                double inner = mesh.calc_sector_angle(vih_it); //inner face angle between halfedge cur and next halfedge
                if(inner < 0) inner += 2.0*M_PI; //because OpenMesh returns negative angles at convex boundary vertices
                angle_accum += inner;
            }

            // now normalize cumulative angles to 2*PI
            double factor = 2.0 * M_PI / angle_accum;
            for (vih_it = mesh.cvih_iter(VertexHandle(v)); vih_it; ++vih_it) {
                h_accumangle_[vih_it.handle().idx()] *= factor;
            }
        }
    }

    // get (flattened) angle between two halfedges at a vertex
    double flat_angle(HalfedgeHandle& in, HalfedgeHandle& out)
    {
        double angle1 = h_accumangle_[in.idx()];
        double angle2 = h_accumangle_[mesh.opposite_halfedge_handle(out).idx()];
        return fmod(angle2 - angle1 + 2.0*M_PI, 2.0*M_PI);
    }

    // Dijkstra update from v along he to w with vector-valued short term memory of length k
    double update_dist(int v, HalfedgeHandle he, int w, int k)
    {
        Vec2d E(0.0, 0.0);
        double current_angle = 0.0;
        double min_dist = std::numeric_limits<double>::infinity();

        // halfedge and vertex currently under consideration
        HalfedgeHandle cur_h(mesh.opposite_halfedge_handle(he));
        VertexHandle cur_v(w);

        for(int i = 1; i <= k; i++) {
            VertexHandle prev_v = mesh.to_vertex_handle(cur_h); // i-th predecessor vertex
            double e_length = (mesh.point(cur_v) - mesh.point(prev_v)).norm();
            e[i-1] = polar_to_xy(e_length, current_angle); // unfolding to the plane
            E += e[i-1];

            EdgeMetric em = metric[mesh.edge_handle(cur_h).idx()];

            // metric tensor information in unfolded configuration
            angles[i-1] = current_angle + em.angle;
            weights[i-1] = em.weight;

            double len = 0.0;
            Vec2d E_normalized = E.normalized();
            for(int j = 0; j < i; ++j)
                len += (e[j]|E_normalized) * length(E_normalized, angles[j], weights[j]);
            min_dist = std::min(min_dist, dist[prev_v.idx()] + std::max(len, 0.0));

            HalfedgeHandle prev_h(pred[prev_v.idx()]);
            if(i == k || !prev_h.is_valid()) // max. depth k or source vertex reached
                break;

            current_angle += (flat_angle(cur_h, prev_h) - M_PI);
            cur_h = prev_h;
            cur_v = prev_v;
        }
        return min_dist;
    }

    // input data:
    const TriMesh &mesh;
    const std::vector<EdgeMetric> &metric;

    // results / Dijkstra variables:
    std::vector<double>         dist;
    std::vector<HalfedgeHandle> pred; // halfedge pointing to the predecessor vertex
    std::vector<bool>           is_final;

    // precomputed 1-ring angles:
    std::vector<double> h_accumangle_; // cumulative angle per halfedge

    // buffers of size k to unfold edge chains (with metric tensors) into plane:
    std::vector<Vec2d>  e;
    std::vector<double> angles;
    std::vector<Vec2d>  weights;

};




#endif // !STVD_H