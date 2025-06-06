#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <vector>
#include <cmath>
#include <mpi.h>

using namespace std;
typedef vector<vector<float>> matrix;

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int nx = 41;
    int ny = 41;
    int nt = 500;
    int nit = 50;
    double dx = 2.0 / (nx - 1);
    double dy = 2.0 / (ny - 1);
    double dt = .01;
    double rho = 1.;
    double nu = .02;

    // Domain decomposition along y
    int local_ny = ny / size;
    int rem = ny % size;
    int start_j = rank * local_ny + min(rank, rem);
    local_ny += (rank < rem) ? 1 : 0;
    int end_j = start_j + local_ny - 1;

    // Add 2 for ghost rows
    matrix u(local_ny + 2, vector<float>(nx));
    matrix v(local_ny + 2, vector<float>(nx));
    matrix p(local_ny + 2, vector<float>(nx));
    matrix b(local_ny + 2, vector<float>(nx));
    matrix un(local_ny + 2, vector<float>(nx));
    matrix vn(local_ny + 2, vector<float>(nx));
    matrix pn(local_ny + 2, vector<float>(nx));

    for (int j = 0; j < local_ny + 2; j++) {
        for (int i = 0; i < nx; i++) {
            u[j][i] = 0;
            v[j][i] = 0;
            p[j][i] = 0;
            b[j][i] = 0;
        }
    }

    ofstream ufile, vfile, pfile;
    if (rank == 0) {
        ufile.open("u.dat");
        vfile.open("v.dat");
        pfile.open("p.dat");
    }

    // Helper lambda for exchanging ghost rows
    auto exchange_ghost_rows = [&](matrix &arr) {
        MPI_Status status;
        // Send up, receive from below
        if (rank != size - 1) {
            MPI_Sendrecv(&arr[local_ny][0], nx, MPI_FLOAT, rank + 1, 0,
                         &arr[local_ny + 1][0], nx, MPI_FLOAT, rank + 1, 1,
                         MPI_COMM_WORLD, &status);
        }
        // Send down, receive from above
        if (rank != 0) {
            MPI_Sendrecv(&arr[1][0], nx, MPI_FLOAT, rank - 1, 1,
                         &arr[0][0], nx, MPI_FLOAT, rank - 1, 0,
                         MPI_COMM_WORLD, &status);
        }
    };

    for (int n = 0; n < nt; n++) {
        // Compute b
        for (int j = 1; j <= local_ny; j++) {
            for (int i = 1; i < nx - 1; i++) {
                b[j][i] = rho * (1 / dt *
                        ((u[j][i+1] - u[j][i-1]) / (2 * dx) + (v[j+1][i] - v[j-1][i]) / (2 * dy)) -
                        pow((u[j][i+1] - u[j][i-1]) / (2 * dx), 2) -
                        2 * ((u[j+1][i] - u[j-1][i]) / (2 * dy) * (v[j][i+1] - v[j][i-1]) / (2 * dx)) -
                        pow((v[j+1][i] - v[j-1][i]) / (2 * dy), 2));
            }
        }
        exchange_ghost_rows(b);

        for (int it = 0; it < nit; it++) {
            for (int j = 0; j < local_ny + 2; j++) {
                for (int i = 0; i < nx; i++) {
                    pn[j][i] = p[j][i];
                }
            }
            exchange_ghost_rows(pn);

            for (int j = 1; j <= local_ny; j++) {
                for (int i = 1; i < nx - 1; i++) {
                    p[j][i] = (pow(dy, 2) * (pn[j][i+1] + pn[j][i-1]) +
                               pow(dx, 2) * (pn[j+1][i] + pn[j-1][i]) -
                               b[j][i] * pow(dx, 2) * pow(dy, 2)) /
                              (2 * (pow(dx, 2) + pow(dy, 2)));
                }
            }
            // Boundary conditions for p (local)
            for (int j = 1; j <= local_ny; j++) {
                p[j][nx-1] = p[j][nx-2];
                p[j][0] = p[j][1];
            }
            // Top and bottom boundaries (global)
            if (rank == 0) {
                for (int i = 0; i < nx; i++) p[1][i] = p[2][i];
            }
            if (rank == size - 1) {
                for (int i = 0; i < nx; i++) p[local_ny][i] = 0;
            }
            exchange_ghost_rows(p);
        }

        for (int j = 0; j < local_ny + 2; j++) {
            for (int i = 0; i < nx; i++) {
                un[j][i] = u[j][i];
                vn[j][i] = v[j][i];
            }
        }
        exchange_ghost_rows(un);
        exchange_ghost_rows(vn);

        for (int j = 1; j <= local_ny; j++) {
            for (int i = 1; i < nx - 1; i++) {
                u[j][i] = un[j][i] - un[j][i] * dt / dx * (un[j][i] - un[j][i - 1])
                                 - un[j][i] * dt / dy * (un[j][i] - un[j - 1][i])
                                 - dt / (2 * rho * dx) * (p[j][i+1] - p[j][i-1])
                                 + nu * dt / pow(dx, 2) * (un[j][i+1] - 2 * un[j][i] + un[j][i-1])
                                 + nu * dt / pow(dy, 2) * (un[j+1][i] - 2 * un[j][i] + un[j-1][i]);
                v[j][i] = vn[j][i] - vn[j][i] * dt / dx * (vn[j][i] - vn[j][i - 1])
                                 - vn[j][i] * dt / dy * (vn[j][i] - vn[j - 1][i])
                                 - dt / (2 * rho * dx) * (p[j+1][i] - p[j-1][i]) 
                                 + nu * dt / pow(dx, 2) * (vn[j][i+1] - 2 * vn[j][i] + vn[j][i-1])
                                 + nu * dt / pow(dy, 2) * (vn[j+1][i] - 2 * vn[j][i] + vn[j-1][i]);
            }
        }
        exchange_ghost_rows(u);
        exchange_ghost_rows(v);

        // Boundary conditions for u, v (local)
        for (int j = 1; j <= local_ny; j++) {
            u[j][0] = 0;
            u[j][nx-1] = 0;
            v[j][0] = 0;
            v[j][nx-1] = 0;
        }
        // Top and bottom boundaries (global)
        if (rank == 0) {
            for (int i = 0; i < nx; i++) {
                u[1][i] = 0;
                v[1][i] = 0;
            }
        }
        if (rank == size - 1) {
            for (int i = 0; i < nx; i++) {
                u[local_ny][i] = 1;
                v[local_ny][i] = 0;
            }
        }

        // Gather and write output every 10 steps
        if (n % 10 == 0) {
            // Gather u, v, p to rank 0
            int *recvcounts = nullptr, *displs = nullptr;
            if (rank == 0) {
                recvcounts = new int[size];
                displs = new int[size];
                int offset = 0;
                for (int r = 0; r < size; r++) {
                    int rows = ny / size + (r < rem ? 1 : 0);
                    recvcounts[r] = rows * nx;
                    displs[r] = offset;
                    offset += rows * nx;
                }
            }
            vector<float> u_send(local_ny * nx), v_send(local_ny * nx), p_send(local_ny * nx);
            for (int j = 1; j <= local_ny; j++) {
                for (int i = 0; i < nx; i++) {
                    u_send[(j-1)*nx + i] = u[j][i];
                    v_send[(j-1)*nx + i] = v[j][i];
                    p_send[(j-1)*nx + i] = p[j][i];
                }
            }
            vector<float> u_all, v_all, p_all;
            if (rank == 0) {
                u_all.resize(nx*ny);
                v_all.resize(nx*ny);
                p_all.resize(nx*ny);
            }
            MPI_Gatherv(u_send.data(), local_ny*nx, MPI_FLOAT,
                        rank == 0 ? u_all.data() : nullptr, recvcounts, displs, MPI_FLOAT, 0, MPI_COMM_WORLD);
            MPI_Gatherv(v_send.data(), local_ny*nx, MPI_FLOAT,
                        rank == 0 ? v_all.data() : nullptr, recvcounts, displs, MPI_FLOAT, 0, MPI_COMM_WORLD);
            MPI_Gatherv(p_send.data(), local_ny*nx, MPI_FLOAT,
                        rank == 0 ? p_all.data() : nullptr, recvcounts, displs, MPI_FLOAT, 0, MPI_COMM_WORLD);
            if (rank == 0) {
                for (int j = 0; j < ny; j++) {
                    for (int i = 0; i < nx; i++) {
                        ufile << u_all[j*nx + i] << " ";
                    }
                }
                ufile << "\n";
                for (int j = 0; j < ny; j++) {
                    for (int i = 0; i < nx; i++) {
                        vfile << v_all[j*nx + i] << " ";
                    }
                }
                vfile << "\n";
                for (int j = 0; j < ny; j++) {
                    for (int i = 0; i < nx; i++) {
                        pfile << p_all[j*nx + i] << " ";
                    }
                }
                pfile << "\n";
            }
            if (rank == 0) {
                delete[] recvcounts;
                delete[] displs;
            }
        }
    }
    if (rank == 0) {
        ufile.close();
        vfile.close();
        pfile.close();
    }
    MPI_Finalize();
    return 0;
}