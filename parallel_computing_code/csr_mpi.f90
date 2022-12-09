PROGRAM csr_mpi
	include 'mpif.h'

	integer myrank, size_Of_Cluster, ierror, tag
	integer p, n, cnt, nnz, m
	real n_dum, h, s, rhs, m_dum
	integer, dimension(:), allocatable :: au,ia,ja,ju
	real*8, dimension(:), allocatable :: u,b,r,t 
	
	call MPI_INIT(ierror)
	call MPI_COMM_SIZE(MPI_COMM_WORLD, size_Of_Cluster, ierror)
	call MPI_COMM_RANK(MPI_COMM_WORLD, myrank, ierror)
	call MPI_FINALIZE(ierror)
	
	!print *, 'Hello World from process: ', p, 'of ', size_Of_Cluster
	p = 3
	n = 9
	
	!nnz = (n-2)*3+3
	
	n_dum = n
	
	m_dum = n/p
	m = m_dum
	
	nnz = m*3
		
	h = 1/n_dum
	s = 7
	rhs = -s*h*h
		
	allocate(au(1:nnz), ia(1:n+1), ja(1:nnz), ju(1:n)) 
	allocate(b(1:n),u(1:n),r(1:n),t(1:n)) 
	
	if (myrank.eq.0) then
		b(1) = 0
		u(1) = 0
		au = 0
		au(1) = 1
		ia(1) = 1
		ja(1) = 1
		ju(1) = 1
		au(2) = -1
		ja(2) = 2
		cnt = 3
		do i = 2, m
			au(cnt) = -1
			au(cnt+1) = 2
			au(cnt+2) = -1
			ju(i) = cnt +1
			ia(i) = cnt
			u(i) = 0
			b(i) = rhs
			ja(cnt) = i-1
			ja(cnt+1) = i
			ja(cnt+2) = i+1
			cnt = cnt + 3
		end do
		if(p==1) then
			cnt = cnt -3
			au(cnt)=1
			ia(m) = cnt
			ja(cnt) = m
			ju(m) = cnt
			ia(m+1) = cnt+1
			u(m) = 1
			b(m) = 1
			iut = 1
			do i =1,m
				t(i) = u(i)
			end do
		end if	
		ia(m+1) = cnt
		cnt = cnt -1
		iut = 1	
	else if(myrank.eq.p-1) then
		cnt = 1
		do i=1, m-1
			au(cnt) = -1
			au(cnt+1) = 2
			au(cnt+2) = -1
			ju(i) = cnt+1
			ia(i) = cnt
			u(i) = 0
			b(i) = rhs
			ja(cnt) = i-1
			ja(cnt+1) = i
			ja(cnt+2) = i+1
			cnt = cnt+3
		end do
		au(cnt) = 1
		ia(m) = cnt
		ja(cnt) = m
		ju(m) = cnt
		ia(m+1) = cnt+1
		u(m) = 1
		b(m) = 1
		iut = 1
	else
		cnt = 1
		do i=1, m
			au(cnt) = -1
			au(cnt+1) = 2
			au(cnt+2) = -1
			ju(i) = cnt+1
			ia(i) = cnt
			u(i) = 0
			b(i) = rhs
			ja(cnt) = i-1
			ja(cnt+1) = i
			ja(cnt+2) = i+1
			cnt = cnt +3
		end do
		ia(m+1) = cnt
		iut = 2
	end if
	print *, myrank, 'haha', ja
END
