/* -*- Mode: C; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*- */
/* vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4 */
/*
 * MDAnalysis --- https://www.mdanalysis.org
 * Copyright (c) 2006-2018 The MDAnalysis Development Team and contributors
 * (see the file AUTHORS for the full list of names)
 *
 * Released under the GNU Public Licence, v2 or any higher version
 *
 * Please cite your use of MDAnalysis in published work:
 *
 * R. J. Gowers, M. Linke, J. Barnoud, T. J. E. Reddy, M. N. Melo, S. L. Seyler,
 * D. L. Dotson, J. Domanski, S. Buchoux, I. M. Kenney, and O. Beckstein.
 * MDAnalysis: A Python package for the rapid analysis of molecular dynamics
 * simulations. In S. Benthall and S. Rostrup editors, Proceedings of the 15th
 * Python in Science Conference, pages 102-109, Austin, TX, 2016. SciPy.
 *
 * N. Michaud-Agrawal, E. J. Denning, T. B. Woolf, and O. Beckstein.
 * MDAnalysis: A Toolkit for the Analysis of Molecular Dynamics Simulations.
 * J. Comput. Chem. 32 (2011), 2319--2327, doi:10.1002/jcc.21787
 */
#ifndef STATIC_ASSERT_H_
#define STATIC_ASSERT_H_

/*
 * This header file provides a custom static_assert(expr, msg) macro if not
 * available, therefore enable static (i.e., compile-time) assertions for
 * C standard < C11 and C++ standard < C++11.
 *
 * Taken in part from
 * http://www.pixelbeat.org/programming/gcc/static_assert.html
 * (GNU All-Permissive License)
 */

// Include appropriate header if required:
#ifndef __cplusplus
    #include <assert.h>
#elif !defined(__cpp_static_assert)
    #include <cassert>
#endif

// Define static_assert() macro if required:
#if !defined(static_assert) && !defined(__cpp_static_assert)
    #if defined(__STDC_VERSION__) && (__STDC_VERSION__  >= 201112L)
        #define static_assert(e, m) _Static_assert(e, m)
    #else
        #define ASSERT_CONCAT_(a, b) a##b
        #define ASSERT_CONCAT(a, b) ASSERT_CONCAT_(a, b)
        // These can't be used after statements in C89
        #ifdef __COUNTER__
            #define static_assert(e, m) \
                    ;enum { ASSERT_CONCAT(static_assert_, __COUNTER__) = \
                            1 / (int) (!!(e)) }
        #else
            #define static_assert(e,m) \
                    ;enum { ASSERT_CONCAT(assert_line_, __LINE__) = \
                            1 / (int) (!!(e)) }
        #endif
    #endif
#endif

#endif /*STATIC_ASSERT_H_*/
