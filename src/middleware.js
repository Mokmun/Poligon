import { clerkMiddleware, createRouteMatcher } from '@clerk/nextjs/server'

// Make sure that the `/api/webhooks/(.*)` route is not protected here
const isProtectedRoute = createRouteMatcher(['/collection(.*)', '/create3D(.*)'])

export default clerkMiddleware(async (auth, req) => {
  const { userId, redirectToSignIn } = await auth()

  if (!userId && isProtectedRoute(req)) {
    // Add custom logic to run before redirecting

    return redirectToSignIn()
  }
})
export const config = {
  matcher: [
    // Skip Next.js internals and all static files, unless found in search params
    '/((?!_next|[^?]*\\.(?:html?|css|js(?!on)|jpe?g|webp|png|gif|svg|ttf|woff2?|ico|csv|docx?|xlsx?|zip|webmanifest)).*)',
    // Always run for API routes
    '/(api|trpc)(.*)',
  ],
}