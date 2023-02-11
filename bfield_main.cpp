// ============================================================================
//   bfield_main.cpp -- FDTD Electromagnetic Field Solver Server
//
//   Copyright 2023 Scott Forbes
//
// This file is part of BField.
// BField is free software: you can redistribute it and/or modify it under the
// terms of the GNU General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your option) any later
// version.
// BField is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
// FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
// details.
// You should have received a copy of the GNU General Public License along
// with BField. If not, see <https://www.gnu.org/licenses/>.
// ============================================================================

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <time.h>
#include <sys/types.h>
#include <sys/socket.h>
#ifdef __APPLE__
#include <sys/event.h>
#else
#include <kqueue/sys/event.h>
#endif
#include <netdb.h>
#include <errno.h>

#include "bfield_server.h"

int pollsStep = 100;    // number of kevent polls to do per timestep
int usPoll;             // useconds between polls for display request
int stepsPoll = 1;      // number of steps per kevent poll, for fastest running
clock_t tstart;         // real time start
int kq;                 // kernel events queue


// ----------------------------------------------------------------------------

void stopSim() {
    printf("Stopping\n");
    fieldEnd();
    clock_t tend = clock();
    double elapsed = (tend - tstart) / (double)CLOCKS_PER_SEC;
    printf("Elapsed = %3.1f seconds, %3.1fns/cell-step\n", elapsed,
            1e9 * elapsed / (osp->step * osp->Ncb.i * osp->Ncb.j * osp->Ncb.k));
    mode = IDLE;
    osp = NULL;
}

// ----------------------------------------------------------------------------
// Execute one server command.

void doCommand(int fd, char* text) {
    char* p = text;
    const int maxArgs = 20;
    char* args[maxArgs];
    Probe* probe;

    // Parse line of text at p as a list of words
    int ai = 0;
    while (ai < maxArgs && *p && *p != '\n') {
        while (*p && *p == ' ')
            p++;
        char* tok = p;
        while (*p && !(*p == ' ' || *p == '\n'))
            p++;
        size_t siz = p - tok;
        args[ai] = new char[siz+1];
        strncpy(args[ai], tok, siz+1);
        args[ai][siz] = '\0';
        ai++;
    }
    if (ai < 1)
        throw new Err("command expected");
    char cmd = args[0][0];
    size_t _ = 0;
    int r = 0;
    char* cwd = 0;

    switch (cmd) {
        case 'A': { // set parameter
            if (ai < 3)
                throw new Err("parameter name & value expected");
            char* name = args[1];
            char* val = args[2];
            if (strcmp(name, "nsteps") == 0)
                nsteps = atoi(val);
            else if (strcmp(name, "units") == 0)
                unit = strcmp(val, "mm") == 0 ? 0.001 : 1.;
            else if (strcmp(name, "usPoll") == 0)
                usPoll = atoi(val);
            //else if (strcmp(name, "verbose") == 0)
            //    verbose = atoi(val);
            _ = write(fd, "A", 1);
            } break;

        case 'B':   // simple mat block
            new MatBlock(args, ai);
            _ = write(fd, "A", 1);
            break;

        case 'C':   // change directory
            if (ai != 2)
                throw new Err("chdir path expected");
            cwd = args[1];
            r = chdir(cwd);
            if (r != 0)
                throw new Err("chdir returned %d", r);
           printf("chdir %s\n", cwd);
            _ = write(fd, "A", 1);
            break;

        case 'D':   // pause/unpause
            _ = write(fd, "A", 1);
            if (mode == PAUSE) {
                printf("Unpausing sim\n");
                mode = RUN;
            } else {
                printf("Pausing sim\n");
                mode = PAUSE;
            }
            break;

        case 'E':   // end sim
            printf("E: end of sim\n");
            _ = write(fd, "A", 1);
            if (mode != IDLE)
                stopSim();
            break;

        case 'F': { // 'Fields' space mat block
            osp = new OuterSpace(args, ai);
            _ = write(fd, "A", 1);
            } break;

        case 'G': { // subspace mat block
            new SubSpace(args, ai);
            _ = write(fd, "A", 1);
            } break;

        case 'H':   // image-stack mat block
            new IMBlock(args, ai);
            _ = write(fd, "A", 1);
            break;

        case 'L':   // image-layer mat block
            new LayerMBlock(args, ai);
            _ = write(fd, "A", 1);
            break;

        case 'M':   // define material
            new Material((const char**)args, ai);
            _ = write(fd, "A", 1);
            break;

        case 'P':   // define probe block
            Probe::add(args, ai);
            _ = write(fd, "A", 1);
            break;

        case 'Q':   // query probe data
            char ack[8];
            if (mode != IDLE) {
                if (ai != 2)
                    throw new Err("probe Q cmd: expected name arg");
                snprintf(ack, 8, "A%06d", osp->step);
                _ = write(fd, ack, 7);
                Probe::writeCells(fd, args[1]);
            } else {
                snprintf(ack, 8, "D000000");
                _ = write(fd, ack, 7);
            }
            break;

        case 'R':   // run simulation
            mode = RUN;
            printf("Initializing run\n");
            _ = write(fd, "A", 1);
            tstart = clock();
            osp->step = 0;
            osp->runInit();
            printf("Running...\n");
            break;

        case 'S':   // define source block
            new Source(args, ai);
            _ = write(fd, "A", 1);
            break;

        case 'U':   // update probe block
            if (ai < 2)
                throw new Err("probe name expected");
            probe = Probe::find(args[1]);
            if (!probe)
                throw new Err("can't find probe %s to update", args[1]);
            probe->update(args, ai);
            probe->initICoords1(osp->extend);
            _ = write(fd, "A", 1);
            break;

        default:
            throw new Err("bad server runmode command '%c'", cmd);
    }

    for (char** p = args; p < args + ai; p++)
        delete[] *p;
}

// ----------------------------------------------------------------------------
// Start a server listening to a port, waiting for connection and then
// dispatching on commands to set up and run FDTD simulations.

void server() {
    int portNo = 50007;
    printf("Starting BField server on port %d\n", portNo);

    struct sockaddr_in servaddr;
    bzero(&servaddr, sizeof(servaddr));
    servaddr.sin_family = AF_INET;
    servaddr.sin_addr.s_addr = htons(INADDR_ANY);
    servaddr.sin_port = htons(portNo);
    int opt = 1;
    int listen_fd = socket(AF_INET, SOCK_STREAM, 0);
    setsockopt(listen_fd, SOL_SOCKET, SO_REUSEADDR, (const char*)&opt,
                 sizeof(int));

    int r = bind(listen_fd, (struct sockaddr *)&servaddr, sizeof(servaddr));
    if (r)
        throw new Err("can't listen on port %d: %s", portNo,
                      strerror(errno));

    listen(listen_fd, 10);

    struct kevent evSet;
    EV_SET(&evSet, listen_fd, EVFILT_READ, EV_ADD, 0, 0, NULL);
    kq = kqueue();
    int rc = kevent(kq, &evSet, 1, NULL, 0, NULL);
    if (rc < 0)
        throw new Err("first kevent returned %d", rc);

    int evtCnt = 0;
    int fd = 0;
    nsteps = 0;     // zero means no limit
    usPoll = 1;     // to insure kevent times out sometime
    mode = IDLE;
    osp = NULL;

    // loop for each event-list request
    while (1) {
        // poll kernel events for any Display request, etc.
        struct kevent evList[32];
        struct timespec timeout;
        timeout.tv_sec = 0;
        timeout.tv_nsec = usPoll * 1000;
        //printf("kevent to=%ldns\n", timeout.tv_nsec);
        int nev = kevent(kq, NULL, 0, evList, 32, &timeout);
        //printf("kevent nev=%d\n", nev);
        if (nev < 0) {
            if (nev != -1)
                throw new Err("kevent returned %d", nev);
            // give it a second chance, for when setting Xcode breakpoint
            nev = kevent(kq, NULL, 0, evList, 32, &timeout);
            if (nev < 0)
                throw new Err("kevent returned %d", nev);
        }
        struct kevent* evp = evList;

        for (int i = 0; i < nev; i++, evp++) {
            //printf("i=%d nev=%d flags=%d\n", i, nev, evp->flags);
            if (evp->flags & EV_EOF) {
                printf("disconnect\n");
                fd = (int)evp->ident;
                EV_SET(&evSet, fd, EVFILT_READ, EV_DELETE, 0, 0, NULL);
                rc = kevent(kq, &evSet, 1, NULL, 0, NULL);
                if (rc < 0)
                    throw new Err("disconnect kevent returned %d", rc);
                mode = IDLE;
            }
            else if (evp->ident == listen_fd) {
                printf("Listening\n");
                fd = accept(listen_fd, (struct sockaddr*)NULL, NULL);
                if (fd == -1)
                    throw new Err("accept failed");
                EV_SET(&evSet, fd, EVFILT_READ, EV_ADD, 0, 0, NULL);
                rc = kevent(kq, &evSet, 1, NULL, 0, NULL);
                if (rc < 0)
                    throw new Err("EV_ADD kevent returned %d", rc);
                setupInit();
            }
            else {
                fd = (int)evp->ident;
                char text[256];
                size_t r = recv(fd, text, sizeof(text)-1, 0);
                text[r] = '\0';
                //if ((osp && osp->verbose > 0) || mode == IDLE)
                if (osp && osp->verbose > 0)
                    printf("cmd: %s\n", text);

                try {
                    doCommand(fd, text);
                } catch (Err* err) {
                    err->report();
                    int n = strlen(err->message);
                    char nack[4];
                    snprintf(nack, 4, "N%02d", n);
                    size_t _ = write(fd, nack, 3);
                    _ = write(fd, err->message, n);
                    if (mode == RUN)
                        stopSim();
                }
            }
        }
        if (mode == RUN) {
            //printf("e=%d/%d ", evtCnt, pollsStep);
            if (evtCnt++ > pollsStep) {
                evtCnt = 0;
                for (int i = 0; i < stepsPoll; i++) {
                    //printf(" %d", step);
                    osp->stepE();
                    osp->stepH();
                    osp->step++;
                    if (nsteps > 0 && osp->step >= nsteps) {
                        printf("step %d, done.\n", osp->step);
                        stopSim();
                        goto done;
                    }
                }
            }
        }
        done: continue;
    }
}

// ----------------------------------------------------------------------------

int main(int argc, const char* argv[]) {
    try {
        server();
    } catch (Err* err) {
        err->report();
        exit(-1);
    }
}
