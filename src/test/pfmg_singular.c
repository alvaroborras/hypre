/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "HYPRE.h"
#include "HYPRE_struct_ls.h"
#include "HYPRE_struct_mv.h"
#include "_hypre_utilities.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define NX 3
#define NY 4
#define NZ 8

static HYPRE_Int
Index3( HYPRE_Int i,
        HYPRE_Int j,
        HYPRE_Int k )
{
   return (k * NY + j) * NX + i;
}

static void
RunPFMG( HYPRE_Real scale,
         HYPRE_Int  symmetric,
         HYPRE_Int  relax_type,
         HYPRE_Int  max_levels,
         HYPRE_Int  dirichlet,
         HYPRE_Int *iterations_ptr,
         HYPRE_Real *residual_ptr )
{
   const HYPRE_Int stencil_size = symmetric ? 4 : 7;
   HYPRE_Int       ilower[3] = {1, 1, 1};
   HYPRE_Int       iupper[3] = {NX, NY, NZ};
   HYPRE_Int       entries[7] = {0, 1, 2, 3, 4, 5, 6};
   HYPRE_Int       sym_offsets[4][3] =
   {
      {-1,  0,  0}, { 0, -1,  0}, { 0,  0, -1}, { 0,  0,  0}
   };
   HYPRE_Int       nonsym_offsets[7][3] =
   {
      {-1,  0,  0}, { 1,  0,  0}, { 0, -1,  0}, { 0,  1,  0},
      { 0,  0, -1}, { 0,  0,  1}, { 0,  0,  0}
   };
   HYPRE_StructGrid    grid;
   HYPRE_StructStencil stencil;
   HYPRE_StructMatrix  A;
   HYPRE_StructVector  b;
   HYPRE_StructVector  x;
   HYPRE_StructSolver  solver;
   HYPRE_Real         *a;
   HYPRE_Real         *b_values;
   HYPRE_Real          hx = 1.0 / (HYPRE_Real) NX;
   HYPRE_Real          hy = 1.0 / (HYPRE_Real) NY;
   HYPRE_Real          hz = 1.0 / (HYPRE_Real) NZ;
   HYPRE_Real          cx = scale * hy * hz / hx;
   HYPRE_Real          cy = scale * hx * hz / hy;
   HYPRE_Real          cz = scale * hx * hy / hz;
   HYPRE_Real          local_sum = 0.0;
   HYPRE_Real          mean;
   HYPRE_Int           rank, num_procs;
   HYPRE_Int           z_start, local_nz, n;
   HYPRE_Int           i, j, k, p, s;

   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   local_nz = NZ / num_procs + (rank < NZ % num_procs);
   z_start = rank * (NZ / num_procs) + hypre_min(rank, NZ % num_procs);
   n = NX * NY * local_nz;
   ilower[2] = z_start + 1;
   iupper[2] = z_start + local_nz;

   a = calloc((size_t) stencil_size * n, sizeof(*a));
   b_values = calloc((size_t) n, sizeof(*b_values));

   for (k = 0; k < local_nz; k++)
   {
      HYPRE_Int global_k = z_start + k;

      for (j = 0; j < NY; j++)
      {
         for (i = 0; i < NX; i++)
         {
            HYPRE_Real west  = i > 0           ? cx : 0.0;
            HYPRE_Real east  = i < NX - 1      ? cx : 0.0;
            HYPRE_Real south = j > 0           ? cy : 0.0;
            HYPRE_Real north = j < NY - 1      ? cy : 0.0;
            HYPRE_Real below = global_k > 0    ? cz : 0.0;
            HYPRE_Real above = global_k < NZ - 1 ? cz : 0.0;
            HYPRE_Real diag;

            p = Index3(i, j, k);
            diag = dirichlet ? -2.0 * (cx + cy + cz) :
                   -(west + east + south + north + below + above);

            if (symmetric)
            {
               a[stencil_size * p]     = west;
               a[stencil_size * p + 1] = south;
               a[stencil_size * p + 2] = below;
               a[stencil_size * p + 3] = diag;
            }
            else
            {
               a[stencil_size * p]     = west;
               a[stencil_size * p + 1] = east;
               a[stencil_size * p + 2] = south;
               a[stencil_size * p + 3] = north;
               a[stencil_size * p + 4] = below;
               a[stencil_size * p + 5] = above;
               a[stencil_size * p + 6] = diag;
            }

            b_values[p] = scale * sin(2.0 * M_PI * ((HYPRE_Real) i + 0.5) / NX);
            local_sum += b_values[p];
         }
      }
   }

   MPI_Allreduce(&local_sum, &mean, 1, HYPRE_MPI_REAL, MPI_SUM, MPI_COMM_WORLD);
   mean /= NX * NY * NZ;
   for (p = 0; p < n; p++)
   {
      b_values[p] -= mean;
   }

   HYPRE_StructGridCreate(MPI_COMM_WORLD, 3, &grid);
   HYPRE_StructGridSetExtents(grid, ilower, iupper);
   HYPRE_StructGridAssemble(grid);

   HYPRE_StructStencilCreate(3, stencil_size, &stencil);
   for (s = 0; s < stencil_size; s++)
   {
      HYPRE_StructStencilSetEntry(stencil, s, symmetric ? sym_offsets[s] : nonsym_offsets[s]);
   }

   HYPRE_StructMatrixCreate(MPI_COMM_WORLD, grid, stencil, &A);
   HYPRE_StructMatrixSetSymmetric(A, symmetric);
   HYPRE_StructMatrixInitialize(A);
   HYPRE_StructMatrixSetBoxValues(A, ilower, iupper, stencil_size, entries, a);
   HYPRE_StructMatrixAssemble(A);

   HYPRE_StructVectorCreate(MPI_COMM_WORLD, grid, &b);
   HYPRE_StructVectorInitialize(b);
   HYPRE_StructVectorSetBoxValues(b, ilower, iupper, b_values);
   HYPRE_StructVectorAssemble(b);

   HYPRE_StructVectorCreate(MPI_COMM_WORLD, grid, &x);
   HYPRE_StructVectorInitialize(x);
   HYPRE_StructVectorSetConstantValues(x, 0.0);
   HYPRE_StructVectorAssemble(x);

   HYPRE_StructPFMGCreate(MPI_COMM_WORLD, &solver);
   HYPRE_StructPFMGSetMaxIter(solver, 200);
   HYPRE_StructPFMGSetTol(solver, 1.0e-10);
   HYPRE_StructPFMGSetRelChange(solver, 1);
   HYPRE_StructPFMGSetRelaxType(solver, relax_type);
   HYPRE_StructPFMGSetLogging(solver, 1);
   if (max_levels)
   {
      HYPRE_StructPFMGSetMaxLevels(solver, max_levels);
   }

   HYPRE_StructPFMGSetup(solver, A, b, x);
   HYPRE_StructPFMGSolve(solver, A, b, x);
   HYPRE_StructPFMGGetNumIterations(solver, iterations_ptr);
   HYPRE_StructPFMGGetFinalRelativeResidualNorm(solver, residual_ptr);

   HYPRE_StructPFMGDestroy(solver);
   HYPRE_StructMatrixDestroy(A);
   HYPRE_StructVectorDestroy(b);
   HYPRE_StructVectorDestroy(x);
   HYPRE_StructStencilDestroy(stencil);
   HYPRE_StructGridDestroy(grid);
   free(a);
   free(b_values);
}

static HYPRE_Int
Check( const char *name,
       HYPRE_Real scale,
       HYPRE_Int  symmetric,
       HYPRE_Int  relax_type,
       HYPRE_Int  max_levels,
       HYPRE_Int  dirichlet )
{
   HYPRE_Int  rank, iterations, failed;
   HYPRE_Real residual;

   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   RunPFMG(scale, symmetric, relax_type, max_levels, dirichlet,
           &iterations, &residual);
   failed = !isfinite((double) residual) || residual > 1.0e-8;
   if (!rank)
   {
      printf("%-28s iterations=%3d residual=%.6e\n", name, iterations, residual);
      if (failed)
      {
         fprintf(stderr, "%s failed to converge\n", name);
      }
   }

   return failed;
}

int
main( int argc, char *argv[] )
{
   HYPRE_Int failures = 0;

   MPI_Init(&argc, &argv);
   HYPRE_Initialize();
   HYPRE_SetMemoryLocation(HYPRE_MEMORY_HOST);
   HYPRE_SetExecutionPolicy(HYPRE_EXEC_HOST);

   failures += Check("Neumann RB default", 1.0, 1, 2, 0, 0);
   failures += Check("Neumann RB scale 1e-12", 1.0e-12, 1, 2, 0, 0);
   failures += Check("Neumann RB scale 1e12", 1.0e12, 1, 2, 0, 0);
   failures += Check("Neumann RB capped", 1.0, 1, 2, 6, 0);
   failures += Check("Neumann Jacobi", 1.0, 1, 1, 0, 0);
   failures += Check("Neumann RB nonsymmetric", 1.0, 0, 2, 0, 0);
   failures += Check("Dirichlet RB", 1.0, 1, 2, 0, 1);

   HYPRE_Finalize();
   MPI_Finalize();

   return failures != 0;
}
