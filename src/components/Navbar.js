import React from 'react';
import Link from 'next/link';
import Image from 'next/image';
import { UserButton, SignedIn, SignedOut, SignInButton, SignUpButton } from '@clerk/nextjs';

const Navbar = () => {
    return (
        <nav className="relative top-0 bg-transparent p-6 w-full z-10">
            <div className="container mx-auto flex justify-between items-center">
                <div className="text-white text-lg font-bold">
                    <Link href='/' className='flex items-center gap-4'>
                        <Image src='/assets/logo.png' alt="logo" width={48} height={48} className='object-contain' />
                        <h1 className='text-purple-500 font-extrabold text-start text-3xl hidden md:block'>Poligon</h1>
                    </Link>
                </div>
                
                <div className="flex space-x-4 md:space-x-12">
                    <SignedIn>
                        <span className="md:border-pink-500 md:border text-white px-4 py-2 rounded-xl tracking-wide hover:bg-pink-600">
                            <Link href='/collection'>
                                <h1 className='hidden md:block'>Collection</h1>
                                <Image 
                                    src='/assets/file.svg'
                                    alt='file'
                                    width={32} 
                                    height={32}
                                    className='md:hidden'
                                />
                            </Link>
                        </span>

                        <UserButton />
                    </SignedIn>
                    <SignedOut>
                        <SignInButton><button className="border-pink-500 border text-white px-4 py-2 rounded-xl hover:bg-pink-600">Sign in</button></SignInButton>
                        <SignUpButton><button className="border-pink-500 border text-white px-4 py-2 rounded-xl hover:bg-pink-600">Sign up</button></SignUpButton>
                    </SignedOut>
                </div>

            </div>
        </nav>
    );
};

export default Navbar;