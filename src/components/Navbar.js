import React from 'react';
import Link from 'next/link';
import Image from 'next/image';
import { UserButton, SignedIn, SignedOut, SignInButton, SignUpButton } from '@clerk/nextjs';

const Navbar = () => {
    return (
        <nav className="bg-transparent p-6 absolute w-full z-10">
            <div className="container mx-auto flex justify-between items-center">
                <div className="text-white text-lg font-bold">
                    <Link href='/' className='flex items-center gap-4'>
                        <Image src='/assets/logo.png' alt="logo" width={48} height={48} className='object-contain' />
                        <h1 className='text-purple-500 font-extrabold text-start text-3xl'>Poligon</h1>
                    </Link>
                </div>
                
                <div className="flex space-x-12">
                <SignedIn>
                    <span className=" text-white px-4 py-2 rounded hover:underline tracking-wide">
                        <Link href='/collection'>
                            Collection
                        </Link>
                    </span>
                    <UserButton />
                </SignedIn>
                <SignedOut>
                    <SignInButton><button className="bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-700">Sign in</button></SignInButton>
                    <SignUpButton><button className="bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-700">Sign up</button></SignUpButton>
                </SignedOut>
                </div>
            </div>
        </nav>
    );
};

export default Navbar;