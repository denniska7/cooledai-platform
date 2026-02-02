/**
 * Formspree configuration for CooledAI forms.
 * Replace your_form_id with the ID from https://formspree.io after signing up.
 */
export const FORMSPREE_FORM_ID =
  process.env.NEXT_PUBLIC_FORMSPREE_FORM_ID || "your_form_id";

export const FORMSPREE_ENDPOINT = `https://formspree.io/f/${FORMSPREE_FORM_ID}`;

export function formspreeSubject(email: string): string {
  return `CooledAI New Beta Request: ${email}`;
}
