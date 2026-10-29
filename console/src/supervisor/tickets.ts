/** The tickets table: `GET /api/tickets` (backend/README.md, "Tickets") and its filters. */

export interface StoredTicket {
  number: string
  department: string
  ward: string
  urgency: string
  location_text: string
  repeat_complaint: boolean
  status: string
  decider_revision: string
  call_id: string
  created_at: string
}

export interface TicketFilter {
  department: string
  ward: string
  urgency: string
}

export const ALL = ''
export const NO_FILTER: TicketFilter = { department: ALL, ward: ALL, urgency: ALL }

export function filterTickets(tickets: StoredTicket[], filter: TicketFilter): StoredTicket[] {
  return tickets.filter(
    (t) =>
      (filter.department === ALL || t.department === filter.department) &&
      (filter.ward === ALL || t.ward === filter.ward) &&
      (filter.urgency === ALL || t.urgency === filter.urgency),
  )
}

/** The newest tickets first. The console token goes along, for a backend that checks it. */
export async function fetchTickets(
  token: string,
  limit = 200,
  get: typeof fetch = fetch,
): Promise<StoredTicket[]> {
  const response = await get(`/api/tickets?limit=${limit}`, {
    headers: { Authorization: `Bearer ${token}` },
  })
  if (!response.ok) throw new Error(`Tickets: HTTP ${response.status}`)
  return (await response.json()) as StoredTicket[]
}
