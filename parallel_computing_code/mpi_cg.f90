PROGRAM hello_world_mpi
include 'mpif.h'

	integer process_Rank, size_Of_Cluster, ierror, tag

	integer n, cnt, nnz, maxit
	real n_dum, h, s, rhs
	real error_sum, error_initial, error_rate, error_th	
	integer, dimension(:), allocatable :: au,ia,ja,ju
	real*8, dimension(:), allocatable :: u,b,r 
			
	n = 800
	nnz = (n-2)*3+3
	
	n_dum = n
	h = 1/n_dum
	s = 7
	rhs = -s*h*h
	
	error_th = 0.0001
	
	allocate(au(1:nnz), ia(1:n+1), ja(1:nnz), ju(1:n)) 
	allocate(b(1:n),u(1:n),r(1:n)) 
	
	b(1)=0
	do i=2,n-1
		b(i)= rhs
	end do
	b(n)=1
	
	do i=1,n
		u(i)=0.0
	end do
	
	maxit = 100000000
	ia = 0
	au = 0
	au(1) = 1
	ia(1) = 1
	ja(1) = 1
	ju(1) = 1
	au(2) = -1
	ja(2) = 2
	cnt = 3 
	do i = 2, n-1
		au(cnt) = -1
		au(cnt+1) = 2
		au(cnt+2) = -1
		ju(i) = cnt+1
		ia(i) = cnt
		ja(cnt) = i-1
		ja(cnt+1) = i
		ja(cnt+2) = i+1
		cnt=cnt+3
	end do
	au(cnt) = 1
	ia(n) = cnt
	ja(cnt) = n
	ju(n) = cnt
	ia(n+1) = cnt+1
	
	call MPI_INIT(ierror)
	call MPI_COMM_SIZE(MPI_COMM_WORLD, size_Of_Cluster, ierror)
	call MPI_COMM_RANK(MPI_COMM_WORLD, process_Rank, ierror)

	print *, 'Hello World from process: ', process_Rank, 'of ', size_Of_Cluster

	call MPI_FINALIZE(ierror)
END PROGRAM

