c code by choi(bcg:test part for mpi_sendrecvreplace)
      program main 
      
      implicit none
      include 'mpif.h'

      integer ::AllocateStatus,n,p,myrank,m,i,j,k,rc,z
      integer ::status(MPI_STATUS_SIZE),tag,ierr
      real,dimension(:,:), Allocatable:: a,b,a0
      real    ::emax,emax0,criterion
      real*8  ::rtc,second,t3,t4
      data second/3.35e-9/
      real    t1(10)
      logical Converged

      call MPI_INIT( ierr)
      call MPI_COMM_RANK(MPI_COMM_WORLD,myrank,ierr)
      call MPI_COMM_SIZE(MPI_COMM_WORLD,p,ierr)
      print *, "Process ", myrank, " of ", p, " is alive"

      if(myrank.eq.0) then
c      print*,'Number of n=?'
c      read(*,*)n
c       t3=rtc()
       n=15000
c      print*,'error criterion=?'
c      read(*,*)criterion
       criterion=0.0001

      endif

      call MPI_BCAST(n,1,MPI_INTEGER,0,MPI_COMM_WORLD,ierr)

      

choi  Compute size of local block

      m=n/p
      if(myrank.lt.(n-p*m))then
      m=m+1
      endif

choi  Allocate local arrays

      Allocate ( a(0:n+1,0:m+1),b(n,m),
     %           stat=AllocateStatus)
      if(AllocateStatus /=0) stop "*** Not enough memory***"

      a=0.0
      b=0.0

      do i=1,m
      b(1,i)=1.0
      a(1,i)=1.0
      enddo

      Converged=.false.
      z=0
c      Do while (.NOT.Converged)
       do 10 k=1,100
	
        IF (myrank.eq.0) then

           do j=2,m
	      do i=2,n-1
	         b(i,j)=0.25*(a(i-1,j)+a(i+1,j)+a(i,j-1)+a(i,j+1))
              enddo
           enddo

        elseif(myrank.eq.p-1) then

           do j=1,m-1
              do i=2,n-1
	         b(i,j)=0.25*(a(i-1,j)+a(i+1,j)+a(i,j-1)+a(i,j+1))
              enddo
           enddo

        else

           do j=1,m
	      do i=2,n-1
	         b(i,j)=0.25*(a(i-1,j)+a(i+1,j)+a(i,j-1)+a(i,j+1))
              enddo
           enddo
        ENDIF


Choi	Convergence test!!

        emax=0.0
	do j=1,m
	   do i=1,n
	   emax=amax1(abs(a(i,j)-b(i,j)),emax)
	   enddo
        enddo

        if (myrank.eq.0) then
c       write(6,*)'error=',emax
        emax0=emax
	if(emax0.lt.criterion) Converged=.True.


	endif

        call MPI_BCAST(Converged,1,MPI_LOGICAL,0,MPI_COMM_WORLD,ierr)

Choi    Substitution of new value!!

        do j=1,m
	do i=1,n
	a(i,j)=b(i,j)
	enddo
	enddo

Choi    Communicate!!

	if(mod(myrank,2).eq.1)then

        call MPI_SEND(B(1,1),n,MPI_REAL,myrank-1,tag,MPI_COMM_WORLD,
     %                ierr)
        call MPI_RECV(A(1,0),n,MPI_REAL,myrank-1,tag,MPI_COMM_WORLD,
     %                status,ierr)

           if(myrank.lt.p-1)then

	   call MPI_SEND(B(1,m),n,MPI_REAL,myrank+1,tag,MPI_COMM_WORLD,
     %                  ierr)
	   call MPI_RECV(A(1,m+1),n,MPI_REAL,myrank+1,tag,MPI_COMM_WORLD,
     %                   status,ierr)
           endif
 
        else  ! myrank is even

	   if(myrank.gt.0) then
           call MPI_RECV(A(1,0),n,MPI_REAL,myrank-1,tag,MPI_COMM_WORLD,
     %                   status,ierr)
	   call MPI_SEND(B(1,1),n,MPI_REAL,myrank-1,tag,MPI_COMM_WORLD,
     %                   ierr)
           endif

	   if(myrank.lt.p-1) then
           call MPI_RECV(A(1,m+1),n,MPI_REAL,myrank+1,tag,
     %                MPI_COMM_WORLD,status,ierr)
	   call MPI_SEND(B(1,m),n,MPI_REAL,myrank+1,tag,MPI_COMM_WORLD,
     %                    ierr)
           endif
        endif
        z=z+1
c	ENDDO
10      continue

        
        if (myrank.eq.0) then
c        t4=rtc()
        write(6,*)'Interation :',z
        write(6,*)'Calculation completed successfully.'
        write(6,*)'n=',n
        write(6,*)'criterion :',criterion
        write(6,*)'error=',emax
	endif

101     call MPI_FINALIZE(rc)

        stop
	end program main

