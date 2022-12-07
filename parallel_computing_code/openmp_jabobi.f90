PROGRAM openmp_jacobi
	use omp_lib

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
	
	time_begin = real(i,kind=8)/real(j,kind=8)
	
	!$omp PARALLEL do private(u0,j1,j2)
	do i=1,n
		j1=ia(i)
		j2=ia(i+1)-1
		u0=b(i)-dot_product(au(j1:j2),u(ja(j1:j2)))
		error_initial = u0*u0 + error_initial
		
		!j1=ia(2)
		!j2=ia(2+1)-1
		!error_initial=abs(b(2)-dot_product(au(j1:j2),u(ja(j1:j2))))
	end do
	!$omp end parallel do
	error_initial = sqrt(error_initial)
	do iter=1, maxit
		!$omp PARALLEL do private(j1,j2,ji,temp1,temp)
		do i=1,n
			j1=ia(i)
			j2=ia(i+1)-1
			ji=ju(i)
			temp1=au(ji)
			temp=dot_product(au(j1:j2),u(ja(j1:j2)))
			r(i)=b(i)-(temp-au(ji)*u(i))
		end do
		!$omp end parallel do
		!$omp PARALLEL do private(ji)
		do i=1,n
			ji=ju(i)
			u(i)=r(i)/au(ji)
		end do
		!$omp end parallel do
		if(mod(iter,10)==0) then
			error_sum = 0
			!$omp PARALLEL do private(u0,j1,j2)
			do i = 1,n
				j1=ia(i)
				j2=ia(i+1)-1
				u0=b(i)-dot_product(au(j1:j2),u(ja(j1:j2)))
				error_sum = u0*u0 + error_sum
				
				!j1=ia(1)
				!j2=ia(1+1)-1
				!error_initial=abs(b(1)-dot_product(au(j1:j2),u(ja(j1:j2))))
			end do
			!$omp end parallel do
			error_sum = sqrt(error_sum)
			error_rate = error_sum/error_initial
			if(error_rate.LT.error_th) exit
		end if
		if(mod(iter,10000)==0) print*, iter, error_sum, error_initial, error_rate, error_th
	end do
	call system_clock(i,j,k)
	time_end=real(i,kind=8)/real(j,kind=8)
	time_cpu=time_end-time_begin
end
